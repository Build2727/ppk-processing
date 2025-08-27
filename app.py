# app.py
import os
import io
import re
import sys
import math
import json
import time
import gzip
import shlex
import queue
import ctypes
import base64
import zipfile
import typing as T
import tempfile
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# Page + global settings
# -------------------------
st.set_page_config(page_title="PPK Processing", layout="wide")

GPST_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
# Current GPST-UTC leap seconds. Good enough for time-matching tolerance purposes.
GPST_MINUS_UTC = 18  # seconds

# -------------------------
# Helpers: files & env
# -------------------------
def prepend_bin_to_path():
    """Make sure ./bin and /app/bin (Streamlit Cloud) are on PATH."""
    candidates = [
        os.path.join(os.getcwd(), "bin"),
        "/app/bin"
    ]
    for d in candidates:
        if os.path.isdir(d):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

def which_rnx2rtkp() -> str:
    """Return path to rnx2rtkp or raise if not found."""
    # prefer bundled
    for candidate in ["./bin/rnx2rtkp", "/app/bin/rnx2rtkp"]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    # fallback to system
    from shutil import which
    p = which("rnx2rtkp")
    if p:
        return p
    raise FileNotFoundError("rnx2rtkp not found. Bundle it in ./bin or install RTKLIB on the system.")

def save_upload(uploaded_file, suffix_hint=".dat") -> str:
    if uploaded_file is None:
        return ""
    # Preserve original suffix if possible
    orig = uploaded_file.name
    _, ext = os.path.splitext(orig)
    suffix = ext if ext else suffix_hint
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.getbuffer())
        return f.name

def has_ext(name: str, exts: T.List[str]) -> bool:
    low = name.lower()
    return any(low.endswith(e) for e in exts)

# -------------------------
# Time parsing
# -------------------------
def parse_time_any(s: str) -> datetime | None:
    """Try common timestamp shapes. Return aware UTC time if possible."""
    s = s.strip()
    if not s:
        return None

    # RTKLIB GPST like '2025/08/20 19:23:45.123'
    m = re.match(r"(\d{4})[/\-](\d{2})[/\-](\d{2})\s+(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", s)
    if m:
        yy, mm, dd, HH, MM, SS = m.groups()
        ss = float(SS)
        return datetime(int(yy), int(mm), int(dd), int(HH), int(MM), int(ss), tzinfo=timezone.utc)

    # ISO variants
    for fmt in [
        "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d %H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
    ]:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass

    # GPST seconds since epoch?
    try:
        v = float(s)
        # interpret as GPST seconds from GPST epoch
        dt = GPST_EPOCH + timedelta(seconds=v)
        return dt
    except Exception:
        pass

    return None

# -------------------------
# RTKLIB PPK
# -------------------------
def run_rnx2rtkp(rover_obs: str, nav_file: str, base_obs: str | None, out_pos: str) -> tuple[bool, str]:
    """Run rnx2rtkp to produce a .pos file."""
    exe = which_rnx2rtkp()
    # Minimal flags; you can extend with a config -k if desired.
    cmd = [exe]
    if base_obs:
        cmd += [rover_obs, base_obs]
    else:
        cmd += [rover_obs]
    cmd += [nav_file, "-o", out_pos]

    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        return (os.path.isfile(out_pos) and os.path.getsize(out_pos) > 0, p.stdout)
    except Exception as e:
        return (False, f"Failed invoking rnx2rtkp: {e}")

# -------------------------
# Parse RTKLIB .pos
# -------------------------
@dataclass
class PosEpoch:
    t: datetime  # UTC
    lat: float
    lon: float
    h: float

def parse_pos_file(path: str) -> list[PosEpoch]:
    """Parse RTKLIB .pos results. Extract UTC time, lat(deg), lon(deg), h(m)."""
    rows: list[PosEpoch] = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("%"):
                continue
            # Typical RTKLIB position lines start with time string; columns vary with options.
            # Heuristic: split and look for lat/lon/height by position
            parts = line.split()
            if len(parts) < 4:
                continue
            # The first two tokens often form the date+time, sometimes it's one token.
            # Try joining 2 tokens then parse:
            ts_try = parts[0]
            if re.match(r"\d{4}[/\-]\d{2}[/\-]\d{2}$", parts[0]) and len(parts) >= 2:
                ts_try = parts[0] + " " + parts[1]
                data_start = 2
            else:
                data_start = 1
            t = parse_time_any(ts_try)
            if t is None:
                # try first token alone
                t = parse_time_any(parts[0])
                data_start = 1
            if t is None:
                continue
            # Most common: lat lon h (deg, deg, m) follow soon after
            if len(parts) >= data_start + 3:
                try:
                    lat = float(parts[data_start + 0])
                    lon = float(parts[data_start + 1])
                    h = float(parts[data_start + 2])
                    rows.append(PosEpoch(t, lat, lon, h))
                except Exception:
                    continue
    return rows

# -------------------------
# Geodesy: ECEF <-> geodetic & ENU
# -------------------------
# WGS84
_A = 6378137.0
_F = 1/298.257223563
_E2 = _F * (2 - _F)

def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    N = _A / math.sqrt(1 - _E2 * sin_lat * sin_lat)
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1 - _E2) + h) * sin_lat
    return np.array([x, y, z], dtype=float)

def ecef_to_geodetic(xyz: np.ndarray) -> tuple[float, float, float]:
    x, y, z = xyz
    lon = math.atan2(y, x)
    p = math.sqrt(x*x + y*y)
    lat = math.atan2(z, p * (1 - _E2))
    # iterate for lat
    for _ in range(5):
        N = _A / math.sqrt(1 - _E2 * math.sin(lat)**2)
        h = p / math.cos(lat) - N
        lat_next = math.atan2(z, p * (1 - _E2 * (N / (N + h))))
        if abs(lat_next - lat) < 1e-12:
            lat = lat_next
            break
        lat = lat_next
    N = _A / math.sqrt(1 - _E2 * math.sin(lat)**2)
    h = p / math.cos(lat) - N
    return (math.degrees(lat), math.degrees(lon), h)

def enu_axes(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors e_hat (East), n_hat (North), u_hat (Up) in ECEF at given geodetic."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    sp, cp = math.sin(lon), math.cos(lon)
    e_hat = np.array([-sp, cp, 0.0], dtype=float)
    n_hat = np.array([-sl*cp, -sl*sp, cl], dtype=float)
    u_hat = np.array([cl*cp, cl*sp, sl], dtype=float)
    return e_hat, n_hat, u_hat

def apply_enu_offset(lat_deg: float, lon_deg: float, h: float, dE: float, dN: float, dU: float) -> tuple[float, float, float]:
    """Return geodetic of APC + ENU offset (meters), applied in ECEF."""
    # base point in ECEF
    xyz = geodetic_to_ecef(lat_deg, lon_deg, h)
    e_hat, n_hat, u_hat = enu_axes(lat_deg, lon_deg)
    d_xyz = dE * e_hat + dN * n_hat + dU * u_hat
    xyz_cam = xyz + d_xyz
    return ecef_to_geodetic(xyz_cam)

# -------------------------
# Events parsing (robust-ish)
# -------------------------
@dataclass
class EventRow:
    t: datetime
    dN: float
    dE: float
    dU: float
    name: str

def try_find_columns(df: pd.DataFrame) -> tuple[str, str, str, str | None]:
    """Return (time_col, n_col, e_col, u_col). Try headers; else fallback to indices 3/4/5 (0-based)."""
    cols = [c.lower() for c in df.columns]
    time_like = ["time", "timestamp", "utc", "gpst", "datetime", "date", "gps_time"]
    n_like = ["n", "north", "dn", "offset_n", "north_m"]
    e_like = ["e", "east", "de", "offset_e", "east_m"]
    u_like = ["u", "up", "du", "offset_u", "alt_off", "z", "dv", "down"]  # if 'down', we'll invert later

    def pick(cands):
        for c in cands:
            if c in cols:
                return df.columns[cols.index(c)]
        return None

    tcol = pick(time_like)
    ncol = pick(n_like)
    ecol = pick(e_like)
    ucol = pick(u_like)

    if tcol and ncol and ecol and ucol:
        return tcol, ncol, ecol, ucol

    # Fallback: assume columns 4/5/6 are N/E/U (1-based) = indices 3/4/5 zero-based
    # Try to build default names
    tcol = tcol or df.columns[0]
    ncol = ncol or df.columns[min(3, len(df.columns)-1)]
    ecol = ecol or df.columns[min(4, len(df.columns)-1)]
    ucol = ucol or df.columns[min(5, len(df.columns)-1)]
    return tcol, ncol, ecol, ucol

def parse_events(events_path: str, delay_sec: float) -> list[EventRow]:
    """Read events; return list with timestamps (UTC) and N/E/U in meters. 
       delay_sec is added to the logged time if your logger is EARLIER than exposure."""
    # Try CSV with auto header; if fails, read as whitespace TSV without header.
    try:
        df = pd.read_csv(events_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(events_path, header=None, delim_whitespace=True)
        # Fabricate column names
        df.columns = [f"col{i+1}" for i in range(df.shape[1])]

    tcol, ncol, ecol, ucol = try_find_columns(df)

    events: list[EventRow] = []
    for _, r in df.iterrows():
        t_raw = str(r[tcol]) if tcol in r else None
        t = parse_time_any(t_raw) if t_raw else None
        if t is None:
            # skip if no time
            continue
        # interpret N/E/U
        try:
            dN = float(r[ncol])
            dE = float(r[ecol])
            dU = float(r[ucol])
        except Exception:
            continue

        # If 'down' was provided instead of 'up', invert (common in some logs)
        if ucol.lower() in ["down", "dv"]:
            dU = -dU

        # Apply camera trigger delay: if logger is EARLIER than exposure, add +delay
        t = t + timedelta(seconds=delay_sec)

        name = str(r[df.columns[0]])  # first col often has image or id
        events.append(EventRow(t, dN, dE, dU, name))
    return events

# -------------------------
# Time matching
# -------------------------
def nearest_epoch(target: datetime, epochs: list[PosEpoch], tol_s: float) -> PosEpoch | None:
    best = None
    best_dt = 1e9
    for ep in epochs:
        d = abs((ep.t - target).total_seconds())
        if d < best_dt:
            best_dt = d
            best = ep
    if best is None:
        return None
    return best if best_dt <= tol_s else None

# -------------------------
# UI: Uploaders (fixed)
# -------------------------
OBS_EXTS = [
    ".obs", ".o", ".rnx",
    ".23o", ".24o", ".25o",
    ".obs.gz", ".o.gz", ".rnx.gz",
    ".23o.gz", ".24o.gz", ".25o.gz",
]
NAV_EXTS = [
    ".nav", ".23n", ".24n", ".25n",
    ".gnav", ".hnav", ".qnav", ".rnav",
    ".nav.gz", ".23n.gz", ".24n.gz", ".25n.gz",
    ".gnav.gz", ".hnav.gz", ".qnav.gz", ".rnav.gz",
]
EVENT_EXTS = [".txt", ".csv", ".log"]

st.sidebar.header("Settings")
time_tol = st.sidebar.number_input("Time match tolerance (s)", 0.01, 5.0, 2.0, 0.01)
cam_delay = st.sidebar.number_input("Camera trigger delay to add (s)", -1.0, 1.0, 0.062, 0.001)
st.sidebar.caption("Positive value means the logger time is earlier than exposure, so we add delay.")

st.header("PPK Processing")

st.subheader("1) Upload inputs")
rover_up = st.file_uploader(
    "Rover observation (RINEX) [.obs, .o, .rnx, .YYo] (gz OK)", type=None
)
base_up = st.file_uploader(
    "Base observation (RINEX, optional)", type=None
)
nav_up = st.file_uploader(
    "Navigation/Ephemeris (RINEX NAV) [.nav, .YYn, .gnav/.hnav/.qnav/.rnav] (gz OK)",
    type=None
)
events_up = st.file_uploader(
    "Events file (with per-exposure offsets N/E/U in meters) [.txt/.csv]", type=None
)

# Prepend PATH for bundled RTKLIB
prepend_bin_to_path()

def ext_check(upl, allowed, label) -> bool:
    if upl is None:
        return False
    if not has_ext(upl.name, allowed):
        st.error(f"{label} extension not recognized for `{upl.name}`. Expected one of: {', '.join(allowed)}")
        return False
    return True

okay = True
if rover_up and not ext_check(rover_up, OBS_EXTS, "Rover"):
    okay = False
if base_up and not ext_check(base_up, OBS_EXTS, "Base"):
    okay = False
if nav_up and not ext_check(nav_up, NAV_EXTS, "NAV"):
    okay = False
if events_up and not ext_check(events_up, EVENT_EXTS, "Events"):
    okay = False

run_btn = st.button("Run PPK")

if run_btn:
    if not (rover_up and nav_up and events_up and okay):
        st.warning("Please upload Rover, NAV, and Events. Base is optional.")
    else:
        with st.spinner("Saving inputs..."):
            rover_path = save_upload(rover_up, ".obs")
            base_path = save_upload(base_up, ".obs") if base_up else ""
            nav_path = save_upload(nav_up, ".nav")
            events_path = save_upload(events_up, ".txt")

        st.success("Files saved. Running RTKLIB rnx2rtkp...")

        out_pos = os.path.join(tempfile.gettempdir(), "solution.pos")
        if os.path.isfile(out_pos):
            try:
                os.remove(out_pos)
            except Exception:
                pass

        ok, log = run_rnx2rtkp(rover_path, nav_path, base_path, out_pos)

        with st.expander("RTKLIB stdout / stderr"):
            st.code(log or "(no output)")

        if not ok:
            st.error("PPK failed or .pos not created.")
            st.stop()

        st.success("PPK completed. Parsing solution and events...")

        epochs = parse_pos_file(out_pos)
        if not epochs:
            st.error("No solution epochs parsed from .pos.")
            st.stop()

        events = parse_events(events_path, delay_sec=cam_delay)
        if not events:
            st.error("No valid events parsed from the events file.")
            st.stop()

        # Build matched & offset-applied CSV
        out_rows = []
        misses = 0
        for ev in events:
            ep = nearest_epoch(ev.t, epochs, time_tol)
            if ep is None:
                misses += 1
                continue
            # Apply ENU offsets (N,E,U) correctly as vectors
            lat2, lon2, h2 = apply_enu_offset(ep.lat, ep.lon, ep.h, dE=ev.dE, dN=ev.dN, dU=ev.dU)
            out_rows.append({
                "name": ev.name,
                "event_time_utc": ev.t.isoformat(),
                "lat_deg": lat2,
                "lon_deg": lon2,
                "ellip_height_m": h2
            })

        if misses:
            st.warning(f"{misses} event(s) could not be matched within ±{time_tol:.3f}s and were skipped.")

        if not out_rows:
            st.error("No events matched — nothing to write.")
            st.stop()

        df_out = pd.DataFrame(out_rows, columns=[
            "name", "event_time_utc", "lat_deg", "lon_deg", "ellip_height_m"
        ])

        st.subheader("2) Results")
        st.dataframe(df_out.head(100), use_container_width=True)

        # Download
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv_bytes, file_name="ppk_events_corrected.csv", mime="text/csv")

        st.success("Done.")




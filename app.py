import os
import io
import sys
import math
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Utility: time parsing
# =========================
from datetime import datetime, timezone, timedelta

TIME_FORMATS = [
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
]

def parse_timestr(s: str) -> Optional[datetime]:
    s = s.strip().replace(",", " ").replace("T", " ")
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None


# =========================
# Geodesy helpers (WGS-84)
# =========================
# Avoid external deps (pyproj). All lengths in meters, angles in radians where noted.

A = 6378137.0                 # semi-major axis
F = 1.0 / 298.257223563
B = A * (1 - F)               # semi-minor axis
E2 = 1 - (B*B)/(A*A)          # first eccentricity squared

def deg2rad(d): return d * math.pi / 180.0
def rad2deg(r): return r * 180.0 / math.pi

def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float) -> np.ndarray:
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    sin_lat = math.sin(lat); cos_lat = math.cos(lat)
    sin_lon = math.sin(lon); cos_lon = math.cos(lon)
    N = A / math.sqrt(1 - E2 * sin_lat * sin_lat)
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1 - E2) + h) * sin_lat
    return np.array([x, y, z], dtype=float)

def ecef_to_geodetic(xyz: np.ndarray) -> Tuple[float,float,float]:
    # Bowring’s method
    x, y, z = xyz
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    theta = math.atan2(z * A, p * B)
    sin_t = math.sin(theta); cos_t = math.cos(theta)
    lat = math.atan2(z + ( (A*A - B*B) / B ) * sin_t**3,
                     p - ( (A*A - B*B) / A ) * cos_t**3)
    sin_lat = math.sin(lat)
    N = A / math.sqrt(1 - E2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - N
    return rad2deg(lat), rad2deg(lon), h

def enu_axes(lat_deg: float, lon_deg: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Return E, N, U unit vectors in ECEF at the given geodetic."""
    lat = deg2rad(lat_deg); lon = deg2rad(lon_deg)
    sin_lat = math.sin(lat); cos_lat = math.cos(lat)
    sin_lon = math.sin(lon); cos_lon = math.cos(lon)
    # East
    e = np.array([-sin_lon,  cos_lon, 0.0], dtype=float)
    # North
    n = np.array([-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat], dtype=float)
    # Up
    u = np.array([ cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat], dtype=float)
    return e, n, u

def apply_neu_offset(apc_lat, apc_lon, apc_h, dN, dE, dU, mode_add: bool) -> Tuple[float,float,float]:
    """
    Convert APC geodetic to ECEF, form Δ in ENU and add/subtract in ECEF,
    convert back to geodetic to get camera center.
    mode_add=True: APC → Camera (add Δ)
    mode_add=False: Camera → APC (subtract Δ, i.e. APC = Cam + Δ ⇒ Cam = APC - Δ)
    """
    e, n, u = enu_axes(apc_lat, apc_lon)
    delta_ecef = dE*e + dN*n + dU*u
    apc_ecef = geodetic_to_ecef(apc_lat, apc_lon, apc_h)
    cam_ecef = apc_ecef + (delta_ecef if mode_add else -delta_ecef)
    return ecef_to_geodetic(cam_ecef)


# =========================
# Data classes
# =========================
@dataclass
class Event:
    t: datetime
    dN: float
    dE: float
    dU: float
    # Raw columns (for provenance)
    raw: List[str]


# =========================
# Parse events (headerless or simple CSV/TSV)
# =========================
def parse_events_file(f: io.BytesIO, delay_sec: float) -> List[Event]:
    """
    Accepts many simple forms:
      - time  dN  dE  dU
      - time, dN, dE, dU
    time = 'YYYY/MM/DD HH:MM:SS(.sss)' or 'YYYY-MM-DD HH:MM:SS(.sss)'
    dN/dE/dU meters (N/E positive, U up).
    """
    text = f.read().decode(errors="ignore").strip().splitlines()
    out = []
    for line in text:
        if not line.strip(): 
            continue
        # Replace commas by spaces and split
        toks = line.strip().replace(",", " ").split()
        # find the timestamp (1 or 2 tokens glued)
        if len(toks) >= 2:
            # Try [0] + [1] as the time
            t = parse_timestr(f"{toks[0]} {toks[1]}")
            rest = toks[2:]
            if t is None:
                # Try only [0]
                t = parse_timestr(toks[0])
                rest = toks[1:]
        else:
            t = parse_timestr(toks[0])
            rest = toks[1:]
        if t is None:
            # last resort: join all until we parse a time
            joined = []
            got = None
            for i in range(1, min(4, len(toks))+1):
                joined = " ".join(toks[:i])
                got = parse_timestr(joined)
                if got is not None:
                    t = got
                    rest = toks[i:]
                    break
        if t is None:
            # skip unparsable line
            continue

        # delay: logger earlier than exposure ⇒ add delay
        t = t + timedelta(seconds=delay_sec)

        # Offsets: tolerate missing or malformed
        dN = float(rest[0]) if len(rest) > 0 and is_floaty(rest[0]) else 0.0
        dE = float(rest[1]) if len(rest) > 1 and is_floaty(rest[1]) else 0.0
        dU = float(rest[2]) if len(rest) > 2 and is_floaty(rest[2]) else 0.0
        out.append(Event(t=t, dN=dN, dE=dE, dU=dU, raw=toks))
    return out

def is_floaty(x: str) -> bool:
    try:
        float(x); return True
    except Exception:
        return False


# =========================
# RTKLIB runner (system binary only)
# =========================
def find_rnx2rtkp() -> str:
    p = shutil.which("rnx2rtkp")
    if p: 
        return p
    # Warn if a bundled binary exists (we will not use it)
    for candidate in ("./bin/rnx2rtkp", "/app/bin/rnx2rtkp"):
        if os.path.isfile(candidate):
            st.sidebar.warning(
                f"Found bundled rnx2rtkp at {candidate} but it will be ignored. "
                "Please remove it (GLIBC mismatch risk)."
            )
    raise FileNotFoundError(
        "rnx2rtkp not found on PATH. On Streamlit Cloud this is provided by packages.txt (rtklib)."
    )

def save_upload(up, dirpath, name):
    path = os.path.join(dirpath, name)
    with open(path, "wb") as g:
        g.write(up.getbuffer())
    return path

def run_rtk(rover_obs, base_obs, nav_file, out_pos, extra_args: Optional[List[str]]=None) -> Tuple[int,str]:
    exe = find_rnx2rtkp()
    cmd = [exe]
    if extra_args:
        cmd += extra_args
    # RTKLIB wants: rover.obs [nav] [base.obs] ...
    # The safest for static PPK: rover, nav, base
    cmd += [rover_obs]
    if nav_file:
        cmd += [nav_file]
    if base_obs:
        cmd += [base_obs]
    cmd += ["-o", out_pos]

    st.write("**Command**")
    st.code(" ".join(cmd))
    try:
        p = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
        )
        return p.returncode, p.stdout
    except Exception as e:
        return 999, f"Failed to run rnx2rtkp: {e}"


# =========================
# Parse RTKLIB .pos
# =========================
def parse_pos(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("%"):
                continue
            parts = line.split()
            # Heuristic: RTKLIB .pos default (time lat lon hgt ...)
            # time might be 2 tokens or 1 joined
            if len(parts) < 4:
                continue
            # Try first two tokens as time
            t = parse_timestr(f"{parts[0]} {parts[1]}")
            idx = 2
            if t is None:
                t = parse_timestr(parts[0])
                idx = 1
            if t is None:
                continue
            # lat lon h
            try:
                lat = float(parts[idx]); lon = float(parts[idx+1]); h = float(parts[idx+2])
            except Exception:
                continue
            # Quality/Q etc. (if present)
            q = parts[idx+3] if len(parts) > idx+3 else ""
            rows.append((t, lat, lon, h, q))
    return pd.DataFrame(rows, columns=["time","lat_deg","lon_deg","h_m","Q"])


# =========================
# Nearest match
# =========================
def nearest_idx(times: np.ndarray, target: datetime) -> int:
    # times is array of np.datetime64; but we have Python datetimes
    # Simpler: linear scan (fast enough for typical sizes).
    best_i = 0
    best_dt = abs((times[0] - target).total_seconds())
    for i in range(1, len(times)):
        dt = abs((times[i] - target).total_seconds())
        if dt < best_dt:
            best_i, best_dt = i, dt
    return best_i


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="PPK Processing with RTKLIB", layout="wide")
st.title("PPK Processing (RTKLIB) — Camera positions")

st.sidebar.header("Inputs")
rov_up = st.sidebar.file_uploader("Rover RINEX OBS", type=["obs","OBS","rnx","o"])
base_up = st.sidebar.file_uploader("Base RINEX OBS (or VRS)", type=["obs","OBS","rnx","o"])
nav_up = st.sidebar.file_uploader("RINEX NAV (e.g., .nav/.n)", type=["nav","NAV","n"])
events_up = st.sidebar.file_uploader("Events file (time, dN, dE, dU)", type=["txt","csv"])

st.sidebar.header("Settings")
time_tol = st.sidebar.number_input(
    "Time match tolerance (s)", min_value=0.01, max_value=5.0, value=2.00, step=0.01, format="%.2f"
)
gps_delay = st.sidebar.number_input(
    "GPS delay (s)", min_value=-1.000, max_value=1.000, value=0.062, step=0.001, format="%.3f",
    help="Positive means the logger timestamp is earlier than the exposure, so we add this delay."
)
offset_mode = st.sidebar.radio(
    "Offset meaning (apply to APC at each event):",
    ("APC → Camera (add Δ)", "Camera → APC (subtract Δ)"),
    index=0
)
mode_add = (offset_mode == "APC → Camera (add Δ)")

run_btn = st.sidebar.button("Run PPK + Build CSV")


# Warn if a bundled rnx2rtkp is still present
for cand in ("./bin/rnx2rtkp", "/app/bin/rnx2rtkp"):
    if os.path.isfile(cand):
        st.sidebar.warning(
            f"Bundled rnx2rtkp found at {cand}. It will NOT be used. "
            "Please remove it from the repo to avoid GLIBC errors."
        )


if run_btn:
    if not (rov_up and events_up):
        st.error("Please provide at least Rover OBS and Events file.")
        st.stop()

    with tempfile.TemporaryDirectory() as td:
        rover_path = save_upload(rov_up, td, "rover.obs")
        base_path = save_upload(base_up, td, "base.obs") if base_up else ""
        nav_path  = save_upload(nav_up,  td, "eph.nav")  if nav_up  else ""

        # Run rnx2rtkp
        pos_path = os.path.join(td, "solution.pos")
        rc, out = run_rtk(rover_path, base_path, nav_path, pos_path, extra_args=None)

        with st.expander("RTKLIB stdout / stderr"):
            st.code(out)

        if rc != 0 or not os.path.isfile(pos_path):
            st.error("PPK failed or .pos not created.")
            st.stop()

        # Parse solution
        df_pos = parse_pos(pos_path)
        if df_pos.empty:
            st.error("Parsed .pos is empty/unreadable.")
            st.stop()

        st.success(f"Read {len(df_pos)} solution epochs from .pos")
        st.dataframe(df_pos.head(), use_container_width=True)

        # Parse events with GPS delay
        events = parse_events_file(events_up, delay_sec=gps_delay)
        if not events:
            st.error("No events parsed from events file.")
            st.stop()
        st.info(f"Parsed {len(events)} events (delay applied: {gps_delay:.3f} s)")

        # Build camera CSV by matching nearest solution then applying NEU offsets in ECEF
        times = df_pos["time"].to_numpy()
        out_rows = []
        for ev in events:
            i = nearest_idx(times, ev.t)
            tsol, lat, lon, h, q = df_pos.iloc[i][["time","lat_deg","lon_deg","h_m","Q"]]
            dt = abs((tsol - ev.t).total_seconds())
            if dt > time_tol:
                # skip if outside tolerance
                continue

            cam_lat, cam_lon, cam_h = apply_neu_offset(lat, lon, h, ev.dN, ev.dE, ev.dU, mode_add=mode_add)

            out_rows.append({
                "time_utc": ev.t.isoformat().replace("+00:00","Z"),
                "match_dt_s": round(dt, 3),
                "apc_lat_deg": lat,
                "apc_lon_deg": lon,
                "apc_h_m":    h,
                "Q": q,
                "dN_m": ev.dN,
                "dE_m": ev.dE,
                "dU_m": ev.dU,
                "cam_lat_deg": cam_lat,
                "cam_lon_deg": cam_lon,
                "cam_h_m": cam_h
            })

        if not out_rows:
            st.error("No events matched within tolerance.")
            st.stop()

        df_cam = pd.DataFrame(out_rows)
        # Output CSV for camera centers only (lat/lon/height) + time; keep a richer version for preview
        csv_camera = df_cam[["time_utc","cam_lat_deg","cam_lon_deg","cam_h_m"]].copy()
        csv_full   = df_cam.copy()

        st.subheader("Camera CSV (lat/lon degrees, height meters)")
        st.dataframe(csv_camera, use_container_width=True)

        # Download
        cam_bytes = csv_camera.to_csv(index=False).encode()
        st.download_button("Download camera_positions.csv", cam_bytes, file_name="camera_positions.csv", mime="text/csv")

        with st.expander("Full matched table (with APC, offsets, quality)"):
            st.dataframe(csv_full, use_container_width=True)
            full_bytes = csv_full.to_csv(index=False).encode()
            st.download_button("Download full_results.csv", full_bytes, file_name="full_results.csv", mime="text/csv")
else:
    st.write("Upload files in the sidebar and click **Run PPK + Build CSV**.")
    st.caption("Tip: On Streamlit Cloud, the system RTKLIB (rnx2rtkp) is provided by `packages.txt` (apt: rtklib).")




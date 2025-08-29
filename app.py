import io
import os
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# --------------------
# Time parsing helpers
# --------------------
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


# --------------------
# Geodesy (WGS-84)
# --------------------
A = 6378137.0
F = 1.0 / 298.257223563
B = A * (1 - F)
E2 = 1 - (B * B) / (A * A)


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


def ecef_to_geodetic(xyz: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = xyz
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    theta = math.atan2(z * A, p * B)
    sin_t = math.sin(theta); cos_t = math.cos(theta)
    lat = math.atan2(
        z + ((A * A - B * B) / B) * sin_t ** 3,
        p - ((A * A - B * B) / A) * cos_t ** 3,
    )
    sin_lat = math.sin(lat)
    N = A / math.sqrt(1 - E2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - N
    return rad2deg(lat), rad2deg(lon), h


def enu_axes(lat_deg: float, lon_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = deg2rad(lat_deg); lon = deg2rad(lon_deg)
    sin_lat = math.sin(lat); cos_lat = math.cos(lat)
    sin_lon = math.sin(lon); cos_lon = math.cos(lon)
    e = np.array([-sin_lon,  cos_lon, 0.0], dtype=float)
    n = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=float)
    u = np.array([ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat], dtype=float)
    return e, n, u


def apply_neu_offset(apc_lat, apc_lon, apc_h, dN, dE, dU, mode_add: bool) -> Tuple[float, float, float]:
    e, n, u = enu_axes(apc_lat, apc_lon)
    delta_ecef = dE * e + dN * n + dU * u
    apc_ecef = geodetic_to_ecef(apc_lat, apc_lon, apc_h)
    cam_ecef = apc_ecef + (delta_ecef if mode_add else -delta_ecef)
    return ecef_to_geodetic(cam_ecef)


# --------------------
# Data classes
# --------------------
@dataclass
class Event:
    t: datetime
    dN: float
    dE: float
    dU: float
    raw: List[str]


# --------------------
# Events parsing
# --------------------
def is_floaty(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def parse_events_file(f: io.BytesIO, delay_sec: float) -> List[Event]:
    text = f.read().decode(errors="ignore").strip().splitlines()
    out: List[Event] = []
    for line in text:
        if not line.strip():
            continue
        toks = line.strip().replace(",", " ").split()

        # detect timestamp (allow 1 or 2 tokens)
        t: Optional[datetime] = None
        rest = toks[1:]
        if len(toks) >= 2:
            t = parse_timestr(f"{toks[0]} {toks[1]}")
            if t is None:
                t = parse_timestr(toks[0])
                rest = toks[1:]
            else:
                rest = toks[2:]
        else:
            t = parse_timestr(toks[0])
            rest = toks[1:]

        if t is None:
            # last resort: try progressive join of tokens
            joined = ""
            for i in range(1, min(4, len(toks)) + 1):
                joined = " ".join(toks[:i])
                t = parse_timestr(joined)
                if t is not None:
                    rest = toks[i:]
                    break

        if t is None:
            continue

        # Apply GPS delay: logger earlier than exposure ⇒ ADD delay
        t = t + timedelta(seconds=delay_sec)

        dN = float(rest[0]) if len(rest) > 0 and is_floaty(rest[0]) else 0.0
        dE = float(rest[1]) if len(rest) > 1 and is_floaty(rest[1]) else 0.0
        dU = float(rest[2]) if len(rest) > 2 and is_floaty(rest[2]) else 0.0
        out.append(Event(t=t, dN=dN, dE=dE, dU=dU, raw=toks))
    return out


# --------------------
# RTKLIB integration
# --------------------
def find_rnx2rtkp() -> Tuple[str, str]:
    """
    Returns (exe_path, used_source)
    used_source: 'PATH' or 'BIN_FALLBACK'
    """
    p = shutil.which("rnx2rtkp")
    if p:
        return p, "PATH"

    # fall back to repo binary if present
    for candidate in ("./bin/rnx2rtkp", "bin/rnx2rtkp", "/app/bin/rnx2rtkp"):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate), "BIN_FALLBACK"

    raise FileNotFoundError(
        "rnx2rtkp not found on PATH and no ./bin/rnx2rtkp fallback present. "
        "On Streamlit Cloud this should come from apt via packages.txt."
    )


def rnx2rtkp_header(exe: str) -> str:
    try:
        p = subprocess.run([exe, "-h"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        return p.stdout.strip()
    except Exception as e:
        return f"(failed to run '{exe} -h'): {e}"


def save_upload(up, dirpath, name):
    path = os.path.join(dirpath, name)
    with open(path, "wb") as g:
        g.write(up.getbuffer())
    return path


def run_rtk(rover_obs, base_obs, nav_file, out_pos, extra_args: Optional[List[str]] = None) -> Tuple[int, str, str]:
    exe, used = find_rnx2rtkp()
    cmd = [exe]
    if extra_args:
        cmd += extra_args

    # RTKLIB typical ordering: rover, nav (optional), base (optional)
    cmd += [rover_obs]
    if nav_file:
        cmd += [nav_file]
    if base_obs:
        cmd += [base_obs]
    cmd += ["-o", out_pos]

    proc_out = ""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        proc_out = p.stdout
        return p.returncode, proc_out, " ".join(cmd)
    except Exception as e:
        return 999, f"Failed to run rnx2rtkp: {e}", " ".join(cmd)


# --------------------
# .pos parsing
# --------------------
def parse_pos(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue

            # try first two tokens as time
            t = parse_timestr(f"{parts[0]} {parts[1]}")
            idx = 2
            if t is None:
                t = parse_timestr(parts[0])
                idx = 1
            if t is None:
                continue

            try:
                lat = float(parts[idx])
                lon = float(parts[idx + 1])
                h = float(parts[idx + 2])
            except Exception:
                continue
            q = parts[idx + 3] if len(parts) > idx + 3 else ""
            rows.append((t, lat, lon, h, q))
    return pd.DataFrame(rows, columns=["time", "lat_deg", "lon_deg", "h_m", "Q"])


# --------------------
# Utility
# --------------------
def nearest_idx(times: List[datetime], target: datetime) -> int:
    best_i = 0
    best_dt = abs((times[0] - target).total_seconds())
    for i in range(1, len(times)):
        dt = abs((times[i] - target).total_seconds())
        if dt < best_dt:
            best_i, best_dt = i, dt
    return best_i


# --------------------
# UI
# --------------------
st.set_page_config(page_title="PPK with RTKLIB", layout="wide")
st.title("PPK Processing (RTKLIB) — Camera positions")
st.caption("Upload files in the sidebar and click **Run**.")

with st.sidebar:
    st.header("Inputs")
    # type=None allows any extension; users can still pick the right files
    rov_up = st.file_uploader("Rover RINEX OBS", type=None)
    base_up = st.file_uploader("Base RINEX OBS (or VRS)", type=None)
    nav_up = st.file_uploader("RINEX NAV (optional)", type=None)
    events_up = st.file_uploader("Events file (time, dN, dE, dU)", type=["txt", "csv"])

    st.header("Settings")
    time_tol = st.number_input(
        "Time match tolerance (s)", min_value=0.01, max_value=5.0, value=2.00, step=0.01, format="%.2f"
    )
    gps_delay = st.number_input(
        "GPS delay (s)", min_value=-1.000, max_value=1.000, value=0.062, step=0.001, format="%.3f",
        help="Positive means the logger timestamp is earlier than the exposure; this delay is ADDED."
    )
    offset_mode = st.radio(
        "Offset meaning (apply to APC at each event):",
        ("APC → Camera (add Δ)", "Camera → APC (subtract Δ)"),
        index=0,
    )
    mode_add = (offset_mode == "APC → Camera (add Δ)")

    st.header("RTKLIB status")
    try:
        exe_path, used_src = find_rnx2rtkp()
        st.success(f"Found rnx2rtkp: `{exe_path}` [{used_src}]")
        st.code(rnx2rtkp_header(exe_path))
        if used_src == "BIN_FALLBACK":
            st.warning(
                "Using bundled ./bin/rnx2rtkp fallback. Prefer system `rtklib` via packages.txt on Cloud "
                "to avoid GLIBC issues."
            )
    except Exception as e:
        st.error(str(e))

run_btn = st.sidebar.button("Run PPK + Build CSV")


# Warn if a bundled rnx2rtkp exists (informational)
for cand in ("./bin/rnx2rtkp", "/app/bin/rnx2rtkp"):
    if os.path.isfile(cand):
        st.sidebar.info(f"Repo binary present: {cand}")


if not run_btn:
    st.stop()

# Pre-flight validation
if not rov_up or not events_up:
    st.error("Please provide at least Rover OBS and Events file.")
    st.stop()

with tempfile.TemporaryDirectory() as td:
    rover_path = save_upload(rov_up, td, "rover.obs")
    base_path = save_upload(base_up, td, "base.obs") if base_up else ""
    nav_path = save_upload(nav_up, td, "eph.nav") if nav_up else ""

    pos_path = os.path.join(td, "solution.pos")
    rc, out_text, cmd_str = run_rtk(rover_path, base_path, nav_path, pos_path, extra_args=None)

    st.subheader("RTKLIB command")
    st.code(cmd_str)

    with st.expander("RTKLIB stdout / stderr"):
        st.code(out_text)

    if rc != 0 or not os.path.isfile(pos_path):
        st.error("PPK failed or `.pos` not created.")
        st.stop()

    df_pos = parse_pos(pos_path)
    if df_pos.empty:
        st.error("Parsed `.pos` is empty or unreadable.")
        st.stop()

    st.success(f"Read {len(df_pos)} solution epochs from .pos")
    st.dataframe(df_pos.head(), use_container_width=True)

    # Parse events with GPS delay (ms precision)
    events = parse_events_file(events_up, delay_sec=gps_delay)
    if not events:
        st.error("No events parsed from the Events file.")
        st.stop()
    st.info(f"Parsed {len(events)} events (delay applied: {gps_delay:.3f} s)")

    # Build camera CSV by nearest match then apply NEU offsets
    times = df_pos["time"].tolist()
    out_rows = []
    for ev in events:
        i = nearest_idx(times, ev.t)
        tsol, lat, lon, h, q = df_pos.iloc[i][["time", "lat_deg", "lon_deg", "h_m", "Q"]]
        dt = abs((tsol - ev.t).total_seconds())
        if dt > time_tol:
            continue

        cam_lat, cam_lon, cam_h = apply_neu_offset(lat, lon, h, ev.dN, ev.dE, ev.dU, mode_add=mode_add)

        out_rows.append({
            "time_utc": ev.t.isoformat().replace("+00:00", "Z"),
            "match_dt_s": round(dt, 3),
            "apc_lat_deg": lat,
            "apc_lon_deg": lon,
            "apc_h_m": h,
            "Q": q,
            "dN_m": ev.dN,
            "dE_m": ev.dE,
            "dU_m": ev.dU,
            "cam_lat_deg": cam_lat,
            "cam_lon_deg": cam_lon,
            "cam_h_m": cam_h,
        })

    if not out_rows:
        st.error("No events matched within tolerance.")
        st.stop()

    df_cam = pd.DataFrame(out_rows)
    csv_camera = df_cam[["time_utc", "cam_lat_deg", "cam_lon_deg", "cam_h_m"]].copy()
    csv_full = df_cam.copy()

    st.subheader("Camera CSV (lat/lon deg, height m)")
    st.dataframe(csv_camera, use_container_width=True)

    cam_bytes = csv_camera.to_csv(index=False).encode()
    st.download_button(
        "Download camera_positions.csv",
        cam_bytes,
        file_name="camera_positions.csv",
        mime="text/csv",
    )

    with st.expander("Full matched table (with APC, offsets, quality)"):
        st.dataframe(csv_full, use_container_width=True)
        full_bytes = csv_full.to_csv(index=False).encode()
        st.download_button(
            "Download full_results.csv",
            full_bytes,
            file_name="full_results.csv",
            mime="text/csv",
        )


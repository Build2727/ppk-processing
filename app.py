import os
import io
import math
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
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
    lat = math.atan2(z + ((A*A - B*B) / B) * sin_t**3,
                     p - ((A*A - B*B) / A) * cos_t**3)
    sin_lat = math.sin(lat)
    N = A / math.sqrt(1 - E2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - N
    return rad2deg(lat), rad2deg(lon), h

def enu_axes(lat_deg: float, lon_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = deg2rad(lat_deg); lon = deg2rad(lon_deg)
    sin_lat = math.sin(lat); cos_lat = math.cos(lat)
    sin_lon = math.sin(lon); cos_lon = math.cos(lon)
    e = np.array([-sin_lon,  cos_lon, 0.0], dtype=float)
    n = np.array([-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat], dtype=float)
    u = np.array([ cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat], dtype=float)
    return e, n, u

def apply_neu_offset(apc_lat, apc_lon, apc_h, dN, dE, dU, mode_add: bool) -> Tuple[float, float, float]:
    e, n, u = enu_axes(apc_lat, apc_lon)
    delta_ecef = dE * e + dN * n + dU * u
    apc_ecef = geodetic_to_ecef(apc_lat, apc_lon, apc_h)
    cam_ecef = apc_ecef + (delta_ecef if mode_add else -delta_ecef)
    return ecef_to_geodetic(cam_ecef)

@dataclass
class Event:
    t: datetime
    dN: float
    dE: float
    dU: float
    raw: List[str]

def is_floaty(x: str) -> bool:
    try:
        float(x); return True
    except Exception:
        return False

def parse_events_file(f, delay_sec: float) -> List[Event]:
    text = f.read().decode(errors="ignore").strip().splitlines()
    out = []
    for line in text:
        if not line.strip():
            continue
        toks = line.strip().replace(",", " ").split()
        if len(toks) >= 2:
            t = parse_timestr(f"{toks[0]} {toks[1]}")
            rest = toks[2:]
            if t is None:
                t = parse_timestr(toks[0]); rest = toks[1:]
        else:
            t = parse_timestr(toks[0]); rest = toks[1:]
        if t is None:
            for i in range(1, min(4, len(toks)) + 1):
                joined = " ".join(toks[:i])
                got = parse_timestr(joined)
                if got is not None:
                    t = got; rest = toks[i:]; break
        if t is None:
            continue
        t = t + timedelta(seconds=delay_sec)
        dN = float(rest[0]) if len(rest) > 0 and is_floaty(rest[0]) else 0.0
        dE = float(rest[1]) if len(rest) > 1 and is_floaty(rest[1]) else 0.0
        dU = float(rest[2]) if len(rest) > 2 and is_floaty(rest[2]) else 0.0
        out.append(Event(t=t, dN=dN, dE=dE, dU=dU, raw=toks))
    return out

def find_rnx2rtkp() -> str:
    candidate = "/usr/bin/rnx2rtkp"
    if os.path.isfile(candidate):
        return candidate
    p = shutil.which("rnx2rtkp")
    if p:
        return p
    raise FileNotFoundError(
        "rnx2rtkp not found. On Streamlit Cloud it should be installed via packages.txt (apt: rtklib). "
        "If running locally, install RTKLIB and ensure rnx2rtkp is in PATH."
    )

def _save_or_decompress(up, dest_path: str) -> str:
    name = (up.name or "").lower()
    data = up.getbuffer()
    if name.endswith(".gz"):
        import gzip
        if dest_path.endswith(".gz"):
            dest_path = dest_path[:-3]
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as gzf:
            payload = gzf.read()
        with open(dest_path, "wb") as f:
            f.write(payload)
        return dest_path
    with open(dest_path, "wb") as f:
        f.write(data)
    return dest_path

def save_upload(up, dirpath, target_name) -> str:
    if not up:
        return ""
    path = os.path.join(dirpath, target_name)
    return _save_or_decompress(up, path)

def run_rtk(rover_obs, base_obs, nav_file, out_pos, extra_args: Optional[List[str]] = None) -> Tuple[int, str]:
    exe = find_rnx2rtkp()
    cmd = [exe]
    if extra_args:
        cmd += extra_args
    cmd += [rover_obs]
    if nav_file:
        cmd += [nav_file]
    if base_obs:
        cmd += [base_obs]
    cmd += ["-o", out_pos]
    st.write("RTKLIB command")
    st.code(" ".join(cmd))
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        return p.returncode, p.stdout
    except Exception as e:
        return 999, f"Failed to run rnx2rtkp: {e}"

def parse_pos(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            t = parse_timestr(f"{parts[0]} {parts[1]}")
            idx = 2
            if t is None:
                t = parse_timestr(parts[0]); idx = 1
            if t is None:
                continue
            try:
                lat = float(parts[idx]); lon = float(parts[idx + 1]); h = float(parts[idx + 2])
            except Exception:
                continue
            q = parts[idx + 3] if len(parts) > idx + 3 else ""
            rows.append((t, lat, lon, h, q))
    return pd.DataFrame(rows, columns=["time", "lat_deg", "lon_deg", "h_m", "Q"])

def nearest_idx(times: List[datetime], target: datetime) -> int:
    best_i = 0
    best_dt = abs((times[0] - target).total_seconds())
    for i in range(1, len(times)):
        dt = abs((times[i] - target).total_seconds())
        if dt < best_dt:
            best_i, best_dt = i, dt
    return best_i

st.set_page_config(page_title="PPK with RTKLIB", layout="wide")
st.title("PPK Processing (RTKLIB) — Camera positions")
st.write("Upload files in the sidebar and click Run.")

st.sidebar.header("Inputs")
rov_up    = st.sidebar.file_uploader("Rover RINEX (OBS)",              type=None)
base_up   = st.sidebar.file_uploader("Base RINEX (OBS or VRS)",        type=None)
nav_up    = st.sidebar.file_uploader("RINEX NAV (e.g., .nav/.n/.brdc)", type=None)
events_up = st.sidebar.file_uploader("Events file (time, dN, dE, dU)", type=["txt", "csv"])

st.sidebar.header("Settings")
time_tol = st.sidebar.number_input("Time match tolerance (s)", min_value=0.01, max_value=5.0, value=2.00, step=0.01, format="%.2f")
gps_delay = st.sidebar.number_input("GPS delay (s)", min_value=-1.000, max_value=1.000, value=0.062, step=0.001, format="%.3f",
                                    help="Logger timestamp earlier than exposure; delay is added to event time.")
offset_mode = st.sidebar.radio("Offset meaning (apply to APC at each event):",
                               ("APC → Camera (add Δ)", "Camera → APC (subtract Δ)"), index=0)
mode_add = (offset_mode == "APC → Camera (add Δ)")

st.sidebar.header("Base override (optional)")
use_base_override = st.sidebar.checkbox("Override base station coordinates (LLH)")
base_lat = base_lon = base_h = 0.0
if use_base_override:
    base_lat = st.sidebar.number_input("Base lat (deg, +N / -S)", value=0.0, step=0.000001, format="%.6f")
    base_lon = st.sidebar.number_input("Base lon (deg, +E / -W)", value=0.0, step=0.000001, format="%.6f")
    base_h   = st.sidebar.number_input("Base ellipsoid height (m)", value=0.0, step=0.01, format="%.2f")

run_btn = st.sidebar.button("Run PPK + Build CSV")

if run_btn:
    if not (rov_up and events_up):
        st.error("Please provide at least Rover RINEX and Events file.")
        st.stop()

    with tempfile.TemporaryDirectory() as td:
        rover_path = save_upload(rov_up, td, "rover.obs")
        base_path  = save_upload(base_up, td, "base.obs") if base_up else ""
        nav_path   = save_upload(nav_up,  td, "eph.nav")  if nav_up  else ""

        extra_args = None
        if use_base_override:
            if not (-90.0 <= base_lat <= 90.0) or not (-180.0 <= base_lon <= 180.0):
                st.error("Base LLH out of range. Latitude must be [-90,90], longitude [-180,180].")
                st.stop()
            opt_lines = [
                "ant1-postype = rinexhead",
                "ant2-postype = llh",
                f"ant2-pos1 = {base_lat:.10f}",
                f"ant2-pos2 = {base_lon:.10f}",
                f"ant2-pos3 = {base_h:.4f}",
            ]
            opt_path = os.path.join(td, "rtklib_override.conf")
            with open(opt_path, "w", encoding="utf-8") as cfg:
                cfg.write("\n".join(opt_lines) + "\n")
            extra_args = ["-k", opt_path]

        pos_path = os.path.join(td, "solution.pos")
        rc, out = run_rtk(rover_path, base_path, nav_path, pos_path, extra_args=extra_args)

        with st.expander("RTKLIB stdout / stderr"):
            st.code(out)

        if rc != 0 or not os.path.isfile(pos_path):
            st.error("PPK failed or .pos not created.")
            st.stop()

        df_pos = parse_pos(pos_path)
        if df_pos.empty:
            st.error("Parsed .pos is empty/unreadable.")
            st.stop()

        st.success(f"Read {len(df_pos)} solution epochs from .pos")
        st.dataframe(df_pos.head(), use_container_width=True)

        events = parse_events_file(events_up, delay_sec=gps_delay)
        if not events:
            st.error("No events parsed from events file.")
            st.stop()
        st.info(f"Parsed {len(events)} events (delay applied: {gps_delay:.3f} s)")

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
                "apc_lat_deg": lat, "apc_lon_deg": lon, "apc_h_m": h, "Q": q,
                "dN_m": ev.dN, "dE_m": ev.dE, "dU_m": ev.dU,
                "cam_lat_deg": cam_lat, "cam_lon_deg": cam_lon, "cam_h_m": cam_h,
            })

        if not out_rows:
            st.error("No events matched within tolerance.")
            st.stop()

        df_cam = pd.DataFrame(out_rows)
        csv_camera = df_cam[["time_utc", "cam_lat_deg", "cam_lon_deg", "cam_h_m"]].copy()
        csv_full   = df_cam.copy()

        st.subheader("Camera CSV (lat/lon degrees, height meters)")
        st.dataframe(csv_camera, use_container_width=True)
        st.download_button("Download camera_positions.csv",
                           csv_camera.to_csv(index=False).encode(),
                           file_name="camera_positions.csv", mime="text/csv")

        with st.expander("Full matched table (with APC, offsets, quality)"):
            st.dataframe(csv_full, use_container_width=True)
            st.download_button("Download full_results.csv",
                               csv_full.to_csv(index=False).encode(),
                               file_name="full_results.csv", mime="text/csv")
else:
    st.caption("On Streamlit Cloud, RTKLIB (rnx2rtkp) is installed via packages.txt (apt: rtklib).")


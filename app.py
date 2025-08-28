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
from datetime import datetime, timezone, timedelta


# =========================
# Time parsing
# =========================
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
A = 6378137.0
F = 1.0 / 298.257223563
B = A * (1 - F)
E2 = 1 - (B*B)/(A*A)

def deg2rad(d): return d * math.pi / 180.0
def rad2deg(r): return r * 180.0 / math.pi

def geodetic_to_ecef(lat_deg, lon_deg, h):
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    N = A / math.sqrt(1 - E2 * sin_lat * sin_lat)
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1 - E2) + h) * sin_lat
    return np.array([x, y, z], dtype=float)

def ecef_to_geodetic(xyz):
    x, y, z = xyz
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    theta = math.atan2(z * A, p * B)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    lat = math.atan2(z + ((A*A - B*B) / B) * sin_t**3,
                     p - ((A*A - B*B) / A) * cos_t**3)
    sin_lat = math.sin(lat)
    N = A / math.sqrt(1 - E2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - N
    return rad2deg(lat), rad2deg(lon), h

def enu_axes(lat_deg, lon_deg):
    lat = deg2rad(lat_deg); lon = deg2rad(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    e = np.array([-sin_lon,  cos_lon, 0.0], dtype=float)
    n = np.array([-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat], dtype=float)
    u = np.array([ cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat], dtype=float)
    return e, n, u

def apply_neu_offset(apc_lat, apc_lon, apc_h, dN, dE, dU, mode_add: bool):
    e, n, u = enu_axes(apc_lat, apc_lon)
    delta_ecef = dE*e + dN*n + dU*u
    apc_ecef = geodetic_to_ecef(apc_lat, apc_lon, apc_h)
    cam_ecef = apc_ecef + (delta_ecef if mode_add else -delta_ecef)
    return ecef_to_geodetic(cam_ecef)


# =========================
# Event structure
# =========================
@dataclass
class Event:
    t: datetime
    dN: float
    dE: float
    dU: float
    raw: List[str]


# =========================
# Parse events file
# =========================
def parse_events_file(f: io.BytesIO, delay_sec: float) -> List[Event]:
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
                t = parse_timestr(toks[0])
                rest = toks[1:]
        else:
            t = parse_timestr(toks[0])
            rest = toks[1:]
        if t is None:
            continue

        # Apply GPS delay
        t = t + timedelta(seconds=delay_sec)

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
# RTKLIB runner
# =========================
def find_rnx2rtkp() -> str:
    p = shutil.which("rnx2rtkp")
    if p: 
        return p
    for candidate in ("./bin/rnx2rtkp", "/app/bin/rnx2rtkp"):
        if os.path.isfile(candidate):
            st.sidebar.warning(
                f"Found bundled rnx2rtkp at {candidate} but it will be ignored. "
                "Please remove it to avoid GLIBC errors."
            )
    raise FileNotFoundError("rnx2rtkp not found on PATH. It should come from apt via packages.txt.")

def save_upload(up, dirpath, name):
    path = os.path.join(dirpath, name)
    with open(path, "wb") as g:
        g.write(up.getbuffer())
    return path

def run_rtk(rover_obs, base_obs, nav_file, out_pos):
    exe = find_rnx2rtkp()
    cmd = [exe, rover_obs]
    if nav_file: cmd.append(nav_file)
    if base_obs: cmd.append(base_obs)
    cmd += ["-o", out_pos]

    st.write("**Command**")
    st.code(" ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    return p.returncode, p.stdout


# =========================
# Parse .pos
# =========================
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
                t = parse_timestr(parts[0])
                idx = 1
            if t is None:
                continue
            try:
                lat = float(parts[idx]); lon = float(parts[idx+1]); h = float(parts[idx+2])
            except Exception:
                continue
            q = parts[idx+3] if len(parts) > idx+3 else ""
            rows.append((t, lat, lon, h, q))
    return pd.DataFrame(rows, columns=["time","lat_deg","lon_deg","h_m","Q"])


# =========================
# Nearest match
# =========================
def nearest_idx(times: np.ndarray, target: datetime) -> int:
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
st.set_page_config(page_title="PPK with RTKLIB", layout="wide")
st.title("PPK Processing (RTKLIB) — Camera positions")

st.sidebar.header("Inputs")
rov_up = st.sidebar.file_uploader("Rover RINEX OBS", type=None)
base_up = st.sidebar.file_uploader("Base RINEX OBS (or VRS)", type=None)
nav_up = st.sidebar.file_uploader("RINEX NAV (e.g., .nav/.n)", type=None)
events_up = st.sidebar.file_uploader("Events file (time, dN, dE, dU)", type=["txt","csv"])

st.sidebar.header("Settings")
time_tol = st.sidebar.number_input("Time match tolerance (s)", min_value=0.01, max_value=5.0, value=2.00, step=0.01, format="%.2f")
gps_delay = st.sidebar.number_input("GPS delay (s)", min_value=-1.000, max_value=1.000, value=0.062, step=0.001, format="%.3f")
offset_mode = st.sidebar.radio("Offset meaning:", ("APC → Camera (add Δ)", "Camera → APC (subtract Δ)"), index=0)
mode_add = (offset_mode == "APC → Camera (add Δ)")

run_btn = st.sidebar.button("Run PPK + Build CSV")


if run_btn:
    if not (rov_up and events_up):
        st.error("Please provide at least Rover OBS and Events file.")
        st.stop()

    with tempfile.TemporaryDirectory() as td:
        rover_path = save_upload(rov_up, td, "rover.obs")
        base_path = save_upload(base_up, td, "base.obs") if base_up else ""
        nav_path  = save_upload(nav_up,  td, "eph.nav")  if nav_up  else ""

        pos_path = os.path.join(td, "solution.pos")
        rc, out = run_rtk(rover_path, base_path, nav_path, pos_path)

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

        times = df_pos["time"].to_numpy()
        out_rows = []
        for ev in events:
            i = nearest_idx(times, ev.t)
            tsol, lat, lon, h, q = df_pos.iloc[i][["time","lat_deg","lon_deg","h_m","Q"]]
            dt = abs((tsol - ev.t).total_seconds())
            if dt > time_tol:
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
        csv_camera = df_cam[["time_utc","cam_lat_deg","cam_lon_deg","cam_h_m"]]
        csv_full   = df_cam.copy()

        st.subheader("Camera CSV (lat/lon degrees, height meters)")
        st.dataframe(csv_camera, use_container_width=True)

        cam_bytes = csv_camera.to_csv(index=False).encode()
        st.download_button("Download camera_positions.csv", cam_bytes, file_name="camera_positions.csv", mime="text/csv")

        with st.expander("Full matched table (with APC, offsets, quality)"):
            st.dataframe(csv_full, use_container_width=True)
            full_bytes = csv_full.to_csv(index=False).encode()
            st.download_button("Download full_results.csv", full_bytes, file_name="full_results.csv", mime="text/csv")
else:
    st.write("Upload files in the sidebar and click **Run PPK + Build CSV**.")


import os
import io
import math
import shutil
import tempfile
import subprocess
import pathlib
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Time parsing
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

# GPST helper
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)

def gpst_from_week_tow(week: int, tow: float) -> datetime:
    return GPS_EPOCH + timedelta(weeks=week, seconds=tow)

# =========================
# Geodesy helpers (WGS-84)
# =========================
A = 6378137.0
F = 1.0 / 298.257223563
B = A * (1 - F)
E2 = 1 - (B * B) / (A * A)

def deg2rad(d): return d * math.pi / 180.0
def rad2deg(r): return r * 180.0 / math.pi

def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float) -> np.ndarray:
    lat = deg2rad(lat_deg); lon = deg2rad(lon_deg)
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
        z + ((A*A - B*B) / B) * sin_t**3,
        p - ((A*A - B*B) / A) * cos_t**3
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
    n = np.array([-sin_lat*cos_lon, -sin_lat*sin_lon,  cos_lat], dtype=float)
    u = np.array([ cos_lat*cos_lon,  cos_lat*sin_lon,  sin_lat], dtype=float)
    return e, n, u

def apply_neu_offset(apc_lat, apc_lon, apc_h, dN, dE, dU, mode_add: bool) -> Tuple[float, float, float]:
    e, n, u = enu_axes(apc_lat, apc_lon)
    delta_ecef = dE * e + dN * n + dU * u
    apc_ecef = geodetic_to_ecef(apc_lat, apc_lon, apc_h)
    cam_ecef = apc_ecef + (delta_ecef if mode_add else -delta_ecef)
    return ecef_to_geodetic(cam_ecef)

# =========================
# Events parsing
# =========================
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

def parse_events_file(f: io.BytesIO, delay_sec: float) -> List[Event]:
    text = f.read().decode(errors="ignore").strip().splitlines()
    out: List[Event] = []
    for line in text:
        if not line.strip():
            continue
        toks = line.strip().replace(",", " ").split()
        t = None; rest = []
        if len(toks) >= 2:
            t = parse_timestr(f"{toks[0]} {toks[1]}")
            rest = toks[2:]
            if t is None:
                t = parse_timestr(toks[0]); rest = toks[1:]
        else:
            t = parse_timestr(toks[0]); rest = toks[1:]
        if t is None:
            for i in range(1, min(4, len(toks)) + 1):
                maybe = " ".join(toks[:i])
                got = parse_timestr(maybe)
                if got is not None:
                    t = got; rest = toks[i:]
                    break
        if t is None:
            continue

        t = t + timedelta(seconds=delay_sec)
        dN = float(rest[0]) if len(rest) > 0 and is_floaty(rest[0]) else 0.0
        dE = float(rest[1]) if len(rest) > 1 and is_floaty(rest[1]) else 0.0
        dU = float(rest[2]) if len(rest) > 2 and is_floaty(rest[2]) else 0.0
        out.append(Event(t=t, dN=dN, dE=dE, dU=dU, raw=toks))
    return out

# =========================
# Ensure rnx2rtkp exists
# =========================
def ensure_rnx2rtkp() -> str:
    p = shutil.which("rnx2rtkp")
    if p:
        return p

    # Absolute locations commonly used on Streamlit Cloud
    candidates = [
        "/usr/bin/rnx2rtkp",
        os.path.expanduser("~/.local/bin/rnx2rtkp"),
        str(pathlib.Path(__file__).resolve().parent / "bin" / "rnx2rtkp"),
        "/app/bin/rnx2rtkp",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            # Prepend its folder to PATH so subprocess keeps working
            d = os.path.dirname(c)
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return c

    # Last resort: build (your existing build-from-source logic)
    here = pathlib.Path(__file__).resolve().parent
    home_bin = pathlib.Path.home() / ".local" / "bin"
    built = home_bin / "rnx2rtkp"
    try:
        home_bin.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            repo = td / "RTKLIB"
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/tomojitakasu/RTKLIB.git", str(repo)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            make_dir = repo / "app" / "rnx2rtkp" / "gcc"
            subprocess.run(["make", "-j"], cwd=str(make_dir), check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            compiled = make_dir / "rnx2rtkp"
            if not compiled.is_file():
                raise FileNotFoundError("RTKLIB build finished but rnx2rtkp not found.")
            shutil.copy2(str(compiled), str(built))
            built.chmod(built.stat().st_mode | 0o111)
    except Exception as e:
        raise FileNotFoundError(
            "Failed to locate or build rnx2rtkp. Ensure packages.txt includes 'rtklib' "
            "or add an executable at ./bin/rnx2rtkp. Error: " + str(e)
        )

    os.environ["PATH"] = f"{str(home_bin)}:{os.environ.get('PATH','')}"
    return str(built)


# =========================
# RTK runner
# =========================
def save_upload(up, dirpath, name):
    path = os.path.join(dirpath, name)
    with open(path, "wb") as g:
        g.write(up.getbuffer())
    return path

def run_rtk(rover_obs, base_obs, nav_file, out_pos, extra_args: Optional[List[str]] = None) -> Tuple[int, str]:
    exe = ensure_rnx2rtkp()
    cmd = [exe]
    if extra_args:
        cmd += extra_args
    cmd += [rover_obs]
    if nav_file:
        cmd += [nav_file]
    if base_obs:
        cmd += [base_obs]
    cmd += ["-o", out_pos]

    st.write("**RTKLIB command**")
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
    if not os.path.exists(path):
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue

            t = parse_timestr(f"{parts[0]} {parts[1]}")
            idx = 2
            if t is None:
                week = None; tow = None
                for i in range(min(4, len(parts) - 1)):
                    try:
                        w = int(float(parts[i]))
                        to = float(parts[i + 1])
                        if 800 <= w <= 4000 and 0.0 <= to < 700000.0:
                            week, tow = w, to
                            break
                    except Exception:
                        pass
                if week is not None and tow is not None:
                    t = gpst_from_week_tow(week, tow)
                    start_i = i + 2
                else:
                    continue
            else:
                start_i = idx

            lat = lon = h = None
            for j in range(start_i, len(parts) - 2):
                try:
                    v1 = float(parts[j])
                    v2 = float(parts[j + 1])
                    v3 = float(parts[j + 2])
                except Exception:
                    continue
                if -90.0 <= v1 <= 90.0 and -180.0 <= v2 <= 180.0:
                    lat, lon, h = v1, v2, v3
                    q_idx = j + 3
                    break

            if lat is None:
                continue

            q = parts[q_idx] if q_idx < len(parts) else ""
            rows.append((t, lat, lon, h, q))

    df = pd.DataFrame(rows, columns=["time", "lat_deg", "lon_deg", "h_m", "Q"])
    if not df.empty:
        df = df.sort_values("time").reset_index(drop=True)
    return df

# =========================
# Nearest match
# =========================
def nearest_idx(times: List[datetime], target: datetime) -> int:
    best_i = 0
    best_dt = abs((times[0] - target).total_seconds())
    for i in range(1, len(times)):
        dt = abs((times[i] - target).total_seconds())
        if dt < best_dt:
            best_i, best_dt = i, dt
    return best_i

# =========================
# UI
# =========================
st.set_page_config(page_title="PPK with RTKLIB", layout="wide")
st.title("PPK Processing (RTKLIB) — Camera positions")
st.write("Upload files in the sidebar and click **Run**.")

st.sidebar.header("Inputs")
rov_up  = st.sidebar.file_uploader("Rover RINEX OBS",            type=None)
base_up = st.sidebar.file_uploader("Base RINEX OBS (or VRS)",    type=None)
nav_up  = st.sidebar.file_uploader("RINEX NAV (optional)",       type=None)
events_up = st.sidebar.file_uploader("Events file (time, dN, dE, dU)", type=["txt", "csv"])

st.sidebar.header("Settings")
time_tol = st.sidebar.number_input("Time match tolerance (s)", min_value=0.01, max_value=5.0,
                                   value=2.00, step=0.01, format="%.2f")
gps_delay = st.sidebar.number_input("GPS delay (s)", min_value=-1.000, max_value=1.000,
                                    value=0.062, step=0.001, format="%.3f",
                                    help="Positive means the logger timestamp is earlier than the exposure (add delay).")
offset_mode = st.sidebar.radio(
    "Offset meaning (apply to APC at each event):",
    ("APC → Camera (add Δ)", "Camera → APC (subtract Δ)"),
    index=0
)
mode_add = (offset_mode == "APC → Camera (add Δ)")

st.sidebar.subheader("RTKLIB status")
try:
    rnx_path_once = ensure_rnx2rtkp()
    st.sidebar.success(f"rnx2rtkp: {rnx_path_once}")
except FileNotFoundError as e:
    st.sidebar.error(str(e))
    st.stop()

run_btn = st.sidebar.button("Run PPK + Build CSV")

if run_btn:
    if not (rov_up and events_up):
        st.error("Please provide at least Rover OBS and Events file.")
        st.stop()

    with tempfile.TemporaryDirectory() as td:
        rover_path = save_upload(rov_up, td, "rover.obs")
        base_path  = save_upload(base_up, td, "base.obs") if base_up else ""
        nav_path   = save_upload(nav_up,  td, "eph.nav")  if nav_up  else ""

        pos_path = os.path.join(td, "solution.pos")
        rc, out = run_rtk(rover_path, base_path, nav_path, pos_path, extra_args=None)

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
    st.caption("Tip: If rnx2rtkp isn’t on PATH, this app will build it from RTKLIB source the first time.")


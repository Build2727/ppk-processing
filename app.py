# app.py
import os
import io
import math
import tempfile
import subprocess
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Streamlit setup + styling
# =========================
st.set_page_config(page_title="PPK Processing (APC→Camera in ECEF)", layout="wide")
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #2e7d32 !important;
        color: white !important;
        border: 1px solid #1b5e20 !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #1b5e20 !important;
        color: white !important;
        border: 1px solid #1b5e20 !important;
    }
    </style>
""", unsafe_allow_html=True)


# =========================
# RTKLIB bootstrap (PATH)
# =========================
def prepend_bin_to_path():
    bin_dirs = [
        os.path.join(os.getcwd(), "bin"),  # local repo ./bin
        "/app/bin",                        # Streamlit Cloud convention
    ]
    for d in bin_dirs:
        if os.path.isdir(d):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

def which_rnx2rtkp() -> Tuple[bool, str]:
    try:
        p = subprocess.run(["rnx2rtkp", "-?"], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        return True, (p.stdout.splitlines()[0] if p.stdout else "rnx2rtkp available")
    except FileNotFoundError:
        return False, "rnx2rtkp not found on PATH"

prepend_bin_to_path()
_rnx_ok, _rnx_msg = which_rnx2rtkp()
with st.sidebar:
    st.header("RTKLIB")
    st.write(_rnx_msg)
    if not _rnx_ok:
        st.warning("Bundle a Linux `rnx2rtkp` at **bin/rnx2rtkp** (executable).")


# =========================
# Helpers: Files + RTK run
# =========================
def save_upload_to_tmp(upload) -> str:
    if upload is None:
        return ""
    _, ext = os.path.splitext(upload.name)
    fd, path = tempfile.mkstemp(prefix="ppk_", suffix=ext or "")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    return path

def run_rnx2rtkp(rover_obs: str, base_obs: str, base_nav: str, out_pos_path: str) -> Tuple[str, str, int]:
    cmd = ["rnx2rtkp", rover_obs, base_obs, base_nav, "-o", out_pos_path]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    except FileNotFoundError:
        return (" ".join(cmd), "ERROR: rnx2rtkp not found on PATH.", 0)

    n_epochs = 0
    if os.path.exists(out_pos_path):
        with open(out_pos_path, "r", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("%"):
                    continue
                n_epochs += 1
    return (" ".join(cmd), out, n_epochs)


# =========================
# Parse RTKLIB .pos
# (expects week, tow, lat, lon, hgt somewhere in lines; robust-ish)
# =========================
def parse_rtklib_pos(pos_path: str) -> pd.DataFrame:
    rows = []
    if not os.path.exists(pos_path):
        return pd.DataFrame(columns=["gps_week", "gps_tow_s", "lat_deg", "lon_deg", "hgt_m"])

    with open(pos_path, "r", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            # Find plausible (lat,lon,hgt)
            lat_idx = lon_idx = hgt_idx = None
            for i in range(len(parts)-2):
                try:
                    a = float(parts[i])
                    b = float(parts[i+1])
                    c = float(parts[i+2])
                except:
                    continue
                if -90 <= a <= 90 and -180 <= b <= 180:
                    lat_idx, lon_idx, hgt_idx = i, i+1, i+2
                    break
            if lat_idx is None:
                continue

            # Find (week,tow) in early columns
            week = tow = None
            for i in range(min(4, len(parts)-1)):
                try:
                    wk = int(float(parts[i]))
                    tw = float(parts[i+1])
                except:
                    continue
                if 800 <= wk <= 4000 and 0 <= tw < 700000:
                    week, tow = wk, tw
                    break
            if week is None:
                continue

            try:
                rows.append((
                    week, tow,
                    float(parts[lat_idx]), float(parts[lon_idx]), float(parts[hgt_idx])
                ))
            except:
                pass

    df = pd.DataFrame(rows, columns=["gps_week", "gps_tow_s", "lat_deg", "lon_deg", "hgt_m"])
    df = df.sort_values(["gps_week", "gps_tow_s"]).reset_index(drop=True)
    return df


# =========================
# Parse Events (TXT/CSV) – no headers
# Columns:
#   1 Img, 2 TOW(s), 3 Week, 4 N(m), 5 E(m), 6 U(m),
#   7 Roll(deg), 8 Pitch(deg), 9 Yaw(deg)
# =========================
def parse_events_file(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["image","gps_week","gps_tow_s","N_m","E_m","U_m",
                                     "Roll_deg","Pitch_deg","Yaw_deg"])

    records = []
    with open(file_path, "r", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            parts = [p.strip() for p in s.replace("\t", ",").split(",")]
            if len(parts) < 9:
                continue
            try:
                img  = parts[0]
                tow  = float(parts[1])
                week = int(float(parts[2]))
                n    = float(parts[3])
                e    = float(parts[4])
                u    = float(parts[5])
                roll = float(parts[6])
                pit  = float(parts[7])
                yaw  = float(parts[8])
            except:
                continue
            records.append((img, week, tow, n, e, u, roll, pit, yaw))

    df = pd.DataFrame(records, columns=[
        "image","gps_week","gps_tow_s","N_m","E_m","U_m","Roll_deg","Pitch_deg","Yaw_deg"
    ])
    return df

# Back-compat alias (fixes your NameError)
def parse_events_no_headers(file_path: str) -> pd.DataFrame:
    return parse_events_file(file_path)


# =========================
# Geodesy helpers (WGS84)
# =========================
WGS84_A = 6378137.0
WGS84_F = 1/298.257223563
WGS84_E2 = WGS84_F*(2 - WGS84_F)

def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat*sin_lat)
    x = (N + h_m) * cos_lat * cos_lon
    y = (N + h_m) * cos_lat * sin_lon
    z = (N*(1 - WGS84_E2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=float)

def ecef_to_geodetic(xyz: np.ndarray) -> Tuple[float, float, float]:
    # Bowring/Newton – adequate here
    x, y, z = xyz
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p*(1 - WGS84_E2))  # initial
    for _ in range(5):
        sin_lat, cos_lat = math.sin(lat), math.cos(lat)
        N = WGS84_A / math.sqrt(1 - WGS84_E2*sin_lat*sin_lat)
        h = p/cos_lat - N
        lat_new = math.atan2(z, p*(1 - WGS84_E2*N/(N + h)))
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    N = WGS84_A / math.sqrt(1 - WGS84_E2*sin_lat*sin_lat)
    h = p/cos_lat - N
    return math.degrees(lat), math.degrees(lon), h

def enu_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    # Columns are unit vectors e_hat, n_hat, u_hat in ECEF
    e_hat = np.array([-sin_lon,            cos_lon,           0.0])
    n_hat = np.array([-sin_lat*cos_lon, -sin_lat*sin_lon,  cos_lat])
    u_hat = np.array([ cos_lat*cos_lon,  cos_lat*sin_lon,  sin_lat])
    return np.column_stack((e_hat, n_hat, u_hat))  # 3x3


# =========================
# Core: match + interpolate + apply NEU offsets
# =========================
def interpolate_ecef_by_time(pos_w: pd.DataFrame, tow: float) -> Tuple[np.ndarray, float]:
    """
    Linear interpolation in ECEF between bracketing epochs.
    Returns (ecef_xyz, used_tow), where used_tow is the clamped/selected epoch time.
    """
    tow_pos = pos_w["gps_tow_s"].to_numpy()
    lat_pos = pos_w["lat_deg"].to_numpy()
    lon_pos = pos_w["lon_deg"].to_numpy()
    hgt_pos = pos_w["hgt_m"].to_numpy()

    # Find insertion point
    idx = np.searchsorted(tow_pos, tow)
    if idx == 0:
        # before first: clamp
        p = geodetic_to_ecef(lat_pos[0], lon_pos[0], hgt_pos[0])
        return p, tow_pos[0]
    if idx >= len(tow_pos):
        # after last: clamp
        p = geodetic_to_ecef(lat_pos[-1], lon_pos[-1], hgt_pos[-1])
        return p, tow_pos[-1]

    t0, t1 = tow_pos[idx-1], tow_pos[idx]
    p0 = geodetic_to_ecef(lat_pos[idx-1], lon_pos[idx-1], hgt_pos[idx-1])
    p1 = geodetic_to_ecef(lat_pos[idx],   lon_pos[idx],   hgt_pos[idx])

    if t1 == t0:
        return p0, t0
    alpha = (tow - t0) / (t1 - t0)
    p = (1 - alpha) * p0 + alpha * p1
    return p, tow


def match_events_to_pos_ecef(
    pos_df: pd.DataFrame,
    events_df: pd.DataFrame,
    tol_s: float,
    exposure_delay_s: float,
    use_interpolation: bool
) -> pd.DataFrame:
    """
    For each event:
      - Adjust event TOW by +exposure_delay_s (logging earlier than exposure).
      - Interpolate (or clamp) APC ECEF at that TOW within same GPS week.
      - Apply NEU offsets (N/E/U positive) APC→camera, in ENU at APC location.
      - Convert camera ECEF → geodetic and write CSV row.
    """
    cols = ["Img", "Lat", "Long", "Alt", "Yaw", "Pitch", "Roll", "X acc", "Y acc", "Z acc"]
    if pos_df.empty or events_df.empty:
        return pd.DataFrame(columns=cols)

    out = []

    for wk in sorted(events_df["gps_week"].unique()):
        pos_w = pos_df[pos_df["gps_week"] == wk]
        if pos_w.empty:
            continue
        pos_w = pos_w.sort_values("gps_tow_s")

        for _, ev in events_df[events_df["gps_week"] == wk].iterrows():
            tow_ev = float(ev["gps_tow_s"]) + float(exposure_delay_s)

            # Quick tolerance check (near any .pos time)
            tow_pos = pos_w["gps_tow_s"].to_numpy()
            if tow_pos.size == 0:
                continue
            i = np.searchsorted(tow_pos, tow_ev)
            cand = []
            if 0 <= i < len(tow_pos): cand.append(abs(tow_pos[i] - tow_ev))
            if i-1 >= 0:               cand.append(abs(tow_pos[i-1] - tow_ev))
            best_dt = min(cand) if cand else None
            if best_dt is None or best_dt > tol_s:
                # No acceptable epoch nearby
                continue

            # Interpolate/clamp APC ECEF at tow_ev
            if use_interpolation:
                p_apc_ecef, used_tow = interpolate_ecef_by_time(pos_w, tow_ev)
                # We also need the geodetic of APC at used_tow for ENU basis
                latlonh = interpolate_ecef_by_time(pos_w, tow_ev)[0]
                lat_deg, lon_deg, h_m = ecef_to_geodetic(latlonh)
            else:
                # Snap to nearest epoch
                idxs = []
                if 0 <= i < len(tow_pos): idxs.append(i)
                if i-1 >= 0:               idxs.append(i-1)
                nearest = min(idxs, key=lambda k: abs(tow_pos[k] - tow_ev))
                lat_deg = float(pos_w.iloc[nearest]["lat_deg"])
                lon_deg = float(pos_w.iloc[nearest]["lon_deg"])
                h_m     = float(pos_w.iloc[nearest]["hgt_m"])
                p_apc_ecef = geodetic_to_ecef(lat_deg, lon_deg, h_m)

            # ENU basis at APC location
            R_enu = enu_basis(lat_deg, lon_deg)  # 3x3 (columns = e,n,u)
            # NEU offsets are APC->camera in meters (E/N/U positive)
            dE = float(ev["E_m"])
            dN = float(ev["N_m"])
            dU = float(ev["U_m"])
            # Vector in ECEF:
            d_ecef = R_enu @ np.array([dE, dN, dU], dtype=float)
            p_cam_ecef = p_apc_ecef + d_ecef

            # Back to geodetic for output
            cam_lat, cam_lon, cam_h = ecef_to_geodetic(p_cam_ecef)

            out.append([
                ev["image"],
                cam_lat, cam_lon, cam_h,
                ev["Yaw_deg"], ev["Pitch_deg"], ev["Roll_deg"],
                0.02, 0.02, 0.03
            ])

    return pd.DataFrame(out, columns=cols)


# =========================
# UI
# =========================
st.title("PPK Processor — ECEF Interpolation + NEU Offsets (APC→Camera)")
st.caption("PPK solve (rnx2rtkp) → interpolate APC in ECEF → apply NEU offsets → EXIF CSV (Yaw, Pitch, Roll)")

colL, colR = st.columns(2)
with colL:
    st.subheader("Rover OBS")
    rover_obs_up = st.file_uploader("Rover RINEX OBS", type=None)
    if rover_obs_up: st.write(f"**Loaded:** `{rover_obs_up.name}`")
with colR:
    st.subheader("Base OBS")
    base_obs_up = st.file_uploader("Base RINEX OBS", type=None)
    if base_obs_up: st.write(f"**Loaded:** `{base_obs_up.name}`")
with colL:
    st.subheader("Base NAV")
    base_nav_up = st.file_uploader("Base RINEX NAV", type=None)
    if base_nav_up: st.write(f"**Loaded:** `{base_nav_up.name}`")
with colR:
    st.subheader("Events (TXT/CSV, no headers)")
    st.caption("Cols: Image, TOW(s), GPS Week, N(m), E(m), U(m), Roll, Pitch, Yaw")
    events_up = st.file_uploader("Events file", type=["txt", "csv"])
    if events_up: st.write(f"**Loaded:** `{events_up.name}`")

st.markdown("---")
run_clicked = st.button("Run PPK Processing", use_container_width=True)

# session state
for k, v in [
    ("ppk_out_path",""),
    ("ppk_cmd",""),
    ("ppk_out_log",""),
    ("pos_df", pd.DataFrame()),
    ("events_df", pd.DataFrame())
]:
    if k not in st.session_state:
        st.session_state[k] = v

if run_clicked:
    rover_path = save_upload_to_tmp(rover_obs_up) if rover_obs_up else ""
    base_obs_path = save_upload_to_tmp(base_obs_up) if base_obs_up else ""
    base_nav_path = save_upload_to_tmp(base_nav_up) if base_nav_up else ""
    events_path   = save_upload_to_tmp(events_up)  if events_up  else ""

    if not (rover_path and base_obs_path and base_nav_path):
        st.error("Please provide Rover OBS, Base OBS, and Base NAV files.")
    else:
        out_pos = os.path.join(tempfile.gettempdir(), "solution.pos")
        cmd, out, n_epochs = run_rnx2rtkp(rover_path, base_obs_path, base_nav_path, out_pos)
        st.session_state["ppk_cmd"] = cmd
        st.session_state["ppk_out_log"] = out

        with st.expander("Command"):
            st.code(cmd)
        with st.expander("RTKLIB stdout / stderr"):
            st.code(out or "(no stdout)")

        if os.path.exists(out_pos) and os.path.getsize(out_pos) > 0:
            st.success(f"PPK complete. **.pos generated** (epochs parsed ≈ {n_epochs}).")
            st.session_state["ppk_out_path"] = out_pos
            with open(out_pos, "rb") as f:
                st.download_button("Download solution.pos", f.read(), "solution.pos", "text/plain")
            pos_df = parse_rtklib_pos(out_pos)
            st.session_state["pos_df"] = pos_df
            st.info(f"Parsed {len(pos_df)} epochs from .pos.")
        else:
            st.error("PPK failed or .pos not created.")
            st.session_state["ppk_out_path"] = ""
            st.session_state["pos_df"] = pd.DataFrame()

        if events_path and os.path.exists(events_path):
            st.session_state["events_df"] = parse_events_file(events_path)
            st.info(f"Events parsed: **{len(st.session_state['events_df'])}** rows.")
        else:
            st.session_state["events_df"] = pd.DataFrame()
            st.warning("Upload an events file to produce the EXIF CSV.")

# Build CSV if both are ready
if not st.session_state["pos_df"].empty and not st.session_state["events_df"].empty:
    st.markdown("---")
    st.subheader("Build EXIF CSV (ECEF interpolation + NEU offsets)")

    c1, c2, c3 = st.columns(3)
    with c1:
        tol = st.slider("Time matching tolerance (seconds)", 0.1, 5.0, 2.0, 0.1)
    with c2:
        exposure_delay = st.number_input(
            "Global exposure delay, seconds (logging earlier than exposure)", value=0.062, step=0.001, format="%.3f"
        )
    with c3:
        use_interp = st.checkbox("Interpolate between .pos epochs in time (ECEF)", value=True)

    out_df = match_events_to_pos_ecef(
        st.session_state["pos_df"],
        st.session_state["events_df"],
        tol_s=tol,
        exposure_delay_s=exposure_delay,
        use_interpolation=use_interp
    )

    if out_df.empty:
        st.warning("No matches within tolerance. Try increasing the tolerance.")
    else:
        st.success(f"Matched **{len(out_df)}** images.")
        st.dataframe(out_df.head(20), use_container_width=True)

        csv_bytes = out_df.to_csv(index=False, float_format="%.7f").encode("utf-8")
        st.download_button("Download EXIF CSV", csv_bytes, "exif_ppk.csv", "text/csv")
else:
    if st.session_state["pos_df"].empty:
        st.info("Run PPK first to create a .pos.")
    elif st.session_state["events_df"].empty:
        st.info("Upload an events file to produce the EXIF CSV.")




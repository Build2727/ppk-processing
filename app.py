# app.py
import os
import math
import tempfile
import subprocess
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import shutil

# ------------------------------
# Page setup + BUTTON COLOR CSS
# ------------------------------
st.set_page_config(page_title="Jamie D PPK Processor", layout="wide")

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #2e7d32 !important; /* Green */
        color: white !important;
        border: 1px solid #1b5e20 !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #1b5e20 !important; /* Darker green */
        color: white !important;
        border: 1px solid #1b5e20 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# RTKLIB bootstrap
# ------------------------------
def prepend_bin_to_path():
    """
    Prefer system rnx2rtkp (installed on Streamlit Cloud via packages.txt),
    and only fall back to repo-local ./bin or /app/bin if not found.
    """
    try:
        subprocess.run(["rnx2rtkp", "-?"], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, check=False)
        return
    except FileNotFoundError:
        pass

    for d in [os.path.join(os.getcwd(), "bin"), "/app/bin"]:
        if os.path.isdir(d):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

def which_rnx2rtkp() -> Tuple[bool, str]:
    try:
        p = subprocess.run(["rnx2rtkp", "-?"], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, check=False)
        first = (p.stdout or "").splitlines()[0] if p.stdout else "rnx2rtkp available"
        return True, first
    except FileNotFoundError:
        return False, "rnx2rtkp not found on PATH"

prepend_bin_to_path()
_rnx_ok, _rnx_msg = which_rnx2rtkp()

# Sidebar status
st.sidebar.header("RTKLIB Status")
st.sidebar.write(_rnx_msg if _rnx_ok else "rnx2rtkp not available")
st.sidebar.write(f"rnx2rtkp path: {shutil.which('rnx2rtkp') or 'not found'}")
if not _rnx_ok:
    st.sidebar.warning("On Streamlit Cloud, ensure 'packages.txt' includes 'rtklib' "
                       "or bundle a Linux binary at bin/rnx2rtkp.")

# ------------------------------
# Geodesy helpers (WGS84)
# ------------------------------
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    X = (N + h_m) * cos_lat * cos_lon
    Y = (N + h_m) * cos_lat * sin_lon
    Z = (N * (1.0 - WGS84_E2) + h_m) * sin_lat
    return X, Y, Z

def ecef_to_geodetic(X: float, Y: float, Z: float):
    # Bowring-like iterative solution on latitude
    lon = math.atan2(Y, X)
    p = math.hypot(X, Y)
    lat = math.atan2(Z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        h = p / math.cos(lat) - N
        lat_new = math.atan2(Z, p * (1.0 - WGS84_E2 * (N / (N + h))))
        if abs(lat_new - lat) < 1e-13:
            lat = lat_new
            break
        lat = lat_new
    sin_lat = math.sin(lat)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - N
    return math.degrees(lat), math.degrees(lon), h

def enu_basis(lat_deg: float, lon_deg: float):
    """Return unit vectors (in ECEF) for local East, North, Up at (lat,lon)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sL = math.sin(lat); cL = math.cos(lat)
    sO = math.sin(lon); cO = math.cos(lon)
    e = (-sO, cO, 0.0)
    n = (-sL * cO, -sL * sO, cL)
    u = (cL * cO, cL * sO, sL)
    return e, n, u

def add_vec(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def scale_vec(v, s):
    return (v[0]*s, v[1]*s, v[2]*s)

def apply_offsets_via_ecef(lat_deg: float, lon_deg: float, h_m: float,
                           dN_m: float, dE_m: float, dU_m: float):
    """
    Apply ENU vector [dE, dN, dU] (meters) at APC to get camera center using ECEF math.
    Returns camera (lat, lon, h).
    """
    X, Y, Z = geodetic_to_ecef(lat_deg, lon_deg, h_m)
    e, n, u = enu_basis(lat_deg, lon_deg)
    d_ecef = add_vec(add_vec(scale_vec(e, dE_m), scale_vec(n, dN_m)), scale_vec(u, dU_m))
    Xc, Yc, Zc = add_vec((X, Y, Z), d_ecef)
    return ecef_to_geodetic(Xc, Yc, Zc)

# ------------------------------
# Helpers
# ------------------------------
def save_upload_to_tmp(upload, suffix: str = "") -> str:
    """Save an uploaded file to a temp file, return its path."""
    if upload is None:
        return ""
    _, ext = os.path.splitext(upload.name)
    if suffix:
        ext = suffix
    fd, path = tempfile.mkstemp(prefix="ppk_", suffix=ext or "")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    return path

def run_rnx2rtkp(rover_obs: str, base_obs: str, base_nav: str, out_pos_path: str) -> Tuple[str, str, int]:
    """
    Run RTKLIB rnx2rtkp to produce a .pos.
    Returns (cmd_str, combined_output, parsed_epochs_count_guess)
    """
    cmd = ["rnx2rtkp", rover_obs, base_obs, base_nav, "-o", out_pos_path]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    except FileNotFoundError:
        out = "ERROR: rnx2rtkp not found on PATH."
        return (" ".join(cmd), out, 0)

    # Quick epoch count
    n_epochs = 0
    if os.path.exists(out_pos_path):
        with open(out_pos_path, "r", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("%"):
                    continue
                n_epochs += 1
    return (" ".join(cmd), out, n_epochs)

def parse_rtklib_pos(pos_path: str) -> pd.DataFrame:
    """
    Parse a standard RTKLIB .pos output; extract GPS week,TOW, lat, lon, hgt.
    Heuristic scan for plausible columns.
    """
    rows = []
    if not os.path.exists(pos_path):
        return pd.DataFrame()

    with open(pos_path, "r", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            try:
                _ = [float(x) for x in parts[:8]]
            except ValueError:
                continue

            # find lat/lon/hgt candidates
            lat_idx = lon_idx = hgt_idx = None
            for i in range(len(parts) - 2):
                try:
                    v = float(parts[i]); v2 = float(parts[i + 1]); v3 = float(parts[i + 2])
                except Exception:
                    continue
                if -90 <= v <= 90 and -180 <= v2 <= 180:
                    lat_idx, lon_idx, hgt_idx = i, i + 1, i + 2
                    break

            # week,tow near start
            week = tow = None
            for i in range(min(4, len(parts) - 1)):
                try:
                    w = int(float(parts[i])); t = float(parts[i + 1])
                    if 800 <= w <= 4000 and 0 <= t < 700000:
                        week, tow = w, t
                        break
                except Exception:
                    continue

            if None not in (lat_idx, lon_idx, hgt_idx, week, tow):
                try:
                    lat = float(parts[lat_idx]); lon = float(parts[lon_idx]); hgt = float(parts[hgt_idx])
                except Exception:
                    continue
                rows.append((week, tow, lat, lon, hgt))

    return pd.DataFrame(rows, columns=["gps_week", "gps_tow_s", "lat_deg", "lon_deg", "hgt_m"])

def parse_events_no_headers(file_path: str) -> pd.DataFrame:
    """
    Parse events TXT/CSV with NO headers. **Assumes meters**.
    Columns used (1..9):
      1: Image
      2: GPS TOW (s)
      3: GPS week
      4: North offset (m)   -> antenna is NORTH of camera by this many meters
      5: West  offset (m)   -> antenna is WEST  of camera by this many meters
      6: Alt   offset (m)   -> antenna is ABOVE camera by this many meters
      7: Roll (deg)
      8: Pitch (deg)
      9: Yaw  (deg)
    """
    if not os.path.exists(file_path):
        return pd.DataFrame()

    with open(file_path, "r", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    recs = []
    for ln in lines:
        parts = [p.strip() for p in ln.replace("\t", ",").split(",")]
        if len(parts) < 9:
            continue
        img = parts[0]
        try:
            tow = float(parts[1])
            week = int(float(parts[2]))
            north_m = float(parts[3])
            west_m  = float(parts[4])
            alt_m   = float(parts[5])
            roll = float(parts[6]); pitch = float(parts[7]); yaw = float(parts[8])
        except Exception:
            continue
        recs.append((img, week, tow, north_m, west_m, alt_m, roll, pitch, yaw))

    return pd.DataFrame(
        recs,
        columns=["image", "gps_week", "gps_tow_s", "North_m", "West_m", "Alt_m",
                 "Roll_deg", "Pitch_deg", "Yaw_deg"]
    )

def match_events_to_pos(pos_df: pd.DataFrame,
                        events_df: pd.DataFrame,
                        tol_s: float = 2.0) -> pd.DataFrame:
    """
    For each event (week,tow), find nearest pos epoch with same week and |dt|<=tol_s.
    Convert APC (antenna) → CAMERA using ENU/ECEF with per-event offsets.
    Output columns for EXIF CSV:
      Img, Lat, Long, Alt, Yaw, Pitch, Roll, X acc, Y acc, Z acc
    """
    if pos_df.empty or events_df.empty:
        return pd.DataFrame(columns=["Img", "Lat", "Long", "Alt", "Yaw", "Pitch", "Roll", "X acc", "Y acc", "Z acc"])

    out_rows = []
    for wk in sorted(events_df["gps_week"].unique()):
        pos_w = pos_df[pos_df["gps_week"] == wk]
        if pos_w.empty:
            continue
        tow_pos = pos_w["gps_tow_s"].to_numpy()
        lat_pos = pos_w["lat_deg"].to_numpy()
        lon_pos = pos_w["lon_deg"].to_numpy()
        hgt_pos = pos_w["hgt_m"].to_numpy()

        events_w = events_df[events_df["gps_week"] == wk]
        for _, ev in events_w.iterrows():
            tow = float(ev["gps_tow_s"])
            idx = np.searchsorted(tow_pos, tow)
            cand = []
            if 0 <= idx < len(tow_pos):
                cand.append(idx)
            if idx - 1 >= 0:
                cand.append(idx - 1)

            best = None; best_dt = None
            for ci in cand:
                dt = abs(tow_pos[ci] - tow)
                if best_dt is None or dt < best_dt:
                    best_dt = dt; best = ci

            if best is None or (best_dt is not None and best_dt > tol_s):
                continue

            base_lat = float(lat_pos[best])
            base_lon = float(lon_pos[best])
            base_h   = float(hgt_pos[best])

            # We are given OFFSETS IN METERS describing ANTENNA relative to CAMERA.
            # To move from Antenna(APC) -> Camera:
            #   camera = antenna + [dE, dN, dU] where
            #   dN = -North, dE = +West (west is negative east), dU = -Alt
            north_m = float(ev["North_m"])
            west_m  = float(ev["West_m"])
            alt_m   = float(ev["Alt_m"])
            dN = -north_m
            dE = +west_m
            dU = -alt_m

            adj_lat, adj_lon, adj_h = apply_offsets_via_ecef(base_lat, base_lon, base_h, dN, dE, dU)

            out_rows.append([
                ev["image"], adj_lat, adj_lon, adj_h,
                ev["Yaw_deg"], ev["Pitch_deg"], ev["Roll_deg"],
                0.02, 0.02, 0.03
            ])

    return pd.DataFrame(
        out_rows,
        columns=["Img", "Lat", "Long", "Alt", "Yaw", "Pitch", "Roll", "X acc", "Y acc", "Z acc"]
    )

# ------------------------------
# UI
# ------------------------------
st.title("Jamie D PPK Processor")
st.caption(
    "PPK solve (RTKLIB rnx2rtkp) → time match → APC→Camera using ENU→ECEF (per-event N/W/Alt meters) "
    "→ EXIF CSV (Yaw, Pitch, Roll)."
)

colL, colR = st.columns(2)

with colL:
    st.subheader("Rover OBS")
    rover_obs_up = st.file_uploader(
        "Drag & drop Rover RINEX OBS (any extension is OK, e.g., .obs, .25o, etc.)",
        type=None,
        help="Any extension is fine; we'll pass it to rnx2rtkp."
    )
    if rover_obs_up:
        st.write(f"**Loaded:** `{rover_obs_up.name}`")

with colR:
    st.subheader("Base OBS")
    base_obs_up = st.file_uploader(
        "Drag & drop Base RINEX OBS (any extension OK, e.g., .obs, .25o, etc.)",
        type=None
    )
    if base_obs_up:
        st.write(f"**Loaded:** `{base_obs_up.name}`")

with colL:
    st.subheader("Base NAV")
    base_nav_up = st.file_uploader(
        "Drag & drop Base RINEX NAV (any extension OK, e.g., .nav, .25n, etc.)",
        type=None
    )
    if base_nav_up:
        st.write(f"**Loaded:** `{base_nav_up.name}`")

with colR:
    st.subheader("Events (TXT/CSV, meters)")
    st.caption(
        "No headers. Columns: Image, TOW(s), GPS week, **North(m)**, **West(m)**, **Alt(m)**, "
        "Roll(deg), Pitch(deg), Yaw(deg). "
        "Offsets describe the **ANTENNA relative to the CAMERA** (N/W/above). "
        "App computes CAMERA = APC + [dE, dN, dU] with dN=−North, dE=+West, dU=−Alt using ENU→ECEF."
    )
    events_up = st.file_uploader(
        "Drag & drop events file (meters)",
        type=["txt", "csv"]
    )
    if events_up:
        st.write(f"**Loaded:** `{events_up.name}`")

st.markdown("---")
run_clicked = st.button("Run PPK Processing", use_container_width=True)

# Scratch state
if "ppk_out_path" not in st.session_state:
    st.session_state["ppk_out_path"] = ""
if "ppk_cmd" not in st.session_state:
    st.session_state["ppk_cmd"] = ""
if "ppk_out_log" not in st.session_state:
    st.session_state["ppk_out_log"] = ""
if "pos_df" not in st.session_state:
    st.session_state["pos_df"] = pd.DataFrame()
if "events_df" not in st.session_state:
    st.session_state["events_df"] = pd.DataFrame()

if run_clicked:
    rover_path = save_upload_to_tmp(rover_obs_up) if rover_obs_up else ""
    base_obs_path = save_upload_to_tmp(base_obs_up) if base_obs_up else ""
    base_nav_path = save_upload_to_tmp(base_nav_up) if base_nav_up else ""
    events_path = save_upload_to_tmp(events_up) if events_up else ""

    if not rover_path or not base_obs_path or not base_nav_path:
        st.error("Please provide Rover OBS, Base OBS, and Base NAV files.")
    else:
        out_pos = os.path.join(tempfile.gettempdir(), "solution.pos")
        cmd, out, n_epochs = run_rnx2rtkp(rover_path, base_obs_path, base_nav_path, out_pos)
        st.session_state["ppk_cmd"] = cmd
        st.session_state["ppk_out_log"] = out

        with st.expander("Command", expanded=False):
            st.code(cmd)
        with st.expander("RTKLIB stdout / stderr", expanded=False):
            st.code(out or "(no stdout)")

        if os.path.exists(out_pos) and os.path.getsize(out_pos) > 0:
            st.success(f"PPK solve complete. **.pos generated** (epochs parsed ≈ {n_epochs}).")
            st.session_state["ppk_out_path"] = out_pos

            with open(out_pos, "rb") as f:
                st.download_button(
                    label="Download solution.pos",
                    data=f.read(),
                    file_name="solution.pos",
                    mime="text/plain"
                )

            pos_df = parse_rtklib_pos(out_pos)
            st.session_state["pos_df"] = pos_df
            st.info(f"Parsed {len(pos_df)} epochs from .pos.")
        else:
            st.error("PPK failed or .pos not created. Check the RTKLIB log above.")
            st.session_state["ppk_out_path"] = ""
            st.session_state["pos_df"] = pd.DataFrame()

        if events_path and os.path.exists(events_path):
            events_df = parse_events_no_headers(events_path)
            st.session_state["events_df"] = events_df
            st.info(f"Events parsed: **{len(events_df)}** rows.")
        else:
            st.session_state["events_df"] = pd.DataFrame()
            st.warning("Upload an events file to produce the EXIF CSV.")

# If we have both, allow building the EXIF CSV
if not st.session_state["pos_df"].empty and not st.session_state["events_df"].empty:
    st.markdown("---")
    st.subheader("Build EXIF CSV (APC→Camera via ENU→ECEF; per-event offsets)")

    tol = st.slider("Time matching tolerance (seconds)", 0.1, 5.0, 2.0, 0.1,
                    help="Max allowed |TOW(event)-TOW(.pos)| for the same GPS week.")

    out_df = match_events_to_pos(st.session_state["pos_df"], st.session_state["events_df"], tol_s=tol)

    if out_df.empty:
        st.warning("No matches within tolerance. Try increasing the tolerance.")
    else:
        st.success(f"Matched **{len(out_df)}** images.")
        st.dataframe(out_df.head(20), use_container_width=True)

        csv_bytes = out_df.to_csv(index=False, float_format="%.7f").encode("utf-8")
        st.download_button(
            label="Download EXIF CSV",
            data=csv_bytes,
            file_name="exif_ppk.csv",
            mime="text/csv"
        )
else:
    if st.session_state["pos_df"].empty:
        st.info("Run PPK first to create a .pos.")
    elif st.session_state["events_df"].empty:
        st.info("Upload an events file to produce the EXIF CSV.")

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
# Page + style
# ------------------------------
st.set_page_config(page_title="Jamie D PPK Processor", layout="wide")
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

# ------------------------------
# RTKLIB bootstrap
# ------------------------------
def prepend_bin_to_path():
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

st.sidebar.header("RTKLIB Status")
st.sidebar.write(_rnx_msg if _rnx_ok else "rnx2rtkp not available")
st.sidebar.write(f"rnx2rtkp path: {shutil.which('rnx2rtkp') or 'not found'}")
if not _rnx_ok:
    st.sidebar.warning("On Streamlit Cloud, ensure 'packages.txt' includes 'rtklib' "
                       "or bundle a Linux binary at bin/rnx2rtkp.")

# ------------------------------
# Geodesy (WGS84)
# ------------------------------
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float):
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    sL, cL = math.sin(lat), math.cos(lat)
    sO, cO = math.sin(lon), math.cos(lon)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sL * sL)
    X = (N + h_m) * cL * cO
    Y = (N + h_m) * cL * sO
    Z = (N * (1.0 - WGS84_E2) + h_m) * sL
    return X, Y, Z

def ecef_to_geodetic(X: float, Y: float, Z: float):
    lon = math.atan2(Y, X)
    p = math.hypot(X, Y)
    lat = math.atan2(Z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sL = math.sin(lat)
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sL * sL)
        h = p / math.cos(lat) - N
        lat_new = math.atan2(Z, p * (1.0 - WGS84_E2 * (N / (N + h))))
        if abs(lat_new - lat) < 1e-13:
            lat = lat_new; break
        lat = lat_new
    sL = math.sin(lat); N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sL * sL)
    h = p / math.cos(lat) - N
    return math.degrees(lat), math.degrees(lon), h

def enu_basis(lat_deg: float, lon_deg: float):
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    sL, cL = math.sin(lat), math.cos(lat)
    sO, cO = math.sin(lon), math.cos(lon)
    e = (-sO, cO, 0.0)
    n = (-sL * cO, -sL * sO, cL)
    u = (cL * cO, cL * sO, sL)
    return e, n, u

def add_vec(a, b):  return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def scale_vec(v, s): return (v[0]*s, v[1]*s, v[2]*s)

def apply_offsets_via_ecef(lat_deg: float, lon_deg: float, h_m: float,
                           dN_m: float, dE_m: float, dU_m: float):
    """
    Apply ENU vector [dE, dN, dU] (meters) at APC to get camera center using ECEF math.
    """
    X, Y, Z = geodetic_to_ecef(lat_deg, lon_deg, h_m)
    e, n, u = enu_basis(lat_deg, lon_deg)
    d_ecef = add_vec(add_vec(scale_vec(e, dE_m), scale_vec(n, dN_m)), scale_vec(u, dU_m))
    Xc, Yc, Zc = add_vec((X, Y, Z), d_ecef)
    return ecef_to_geodetic(Xc, Yc, Zc), (X, Y, Z), (Xc, Yc, Zc)

def ecef_delta_to_enu(lat_deg: float, lon_deg: float, X1, Y1, Z1, X2, Y2, Z2):
    """Return ENU of (point2 - point1) expressed at (lat,lon)."""
    e, n, u = enu_basis(lat_deg, lon_deg)
    d = (X2 - X1, Y2 - Y1, Z2 - Z1)
    dE = d[0]*e[0] + d[1]*e[1] + d[2]*e[2]
    dN = d[0]*n[0] + d[1]*n[1] + d[2]*n[2]
    dU = d[0]*u[0] + d[1]*u[1] + d[2]*u[2]
    return dE, dN, dU

# ------------------------------
# IO helpers
# ------------------------------
def save_upload_to_tmp(upload, suffix: str = "") -> str:
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
    cmd = ["rnx2rtkp", rover_obs, base_obs, base_nav, "-o", out_pos_path]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    except FileNotFoundError:
        out = "ERROR: rnx2rtkp not found on PATH."
        return (" ".join(cmd), out, 0)
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
            lat_idx = lon_idx = hgt_idx = None
            for i in range(len(parts) - 2):
                try:
                    v = float(parts[i]); v2 = float(parts[i + 1]); v3 = float(parts[i + 2])
                except Exception:
                    continue
                if -90 <= v <= 90 and -180 <= v2 <= 180:
                    lat_idx, lon_idx, hgt_idx = i, i + 1, i + 2
                    break
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
    Events TXT/CSV (no headers). Assumes meters and N/E/U positive.
    Columns used (1..9):
      1: Image
      2: GPS TOW (s)
      3: GPS week
      4: North offset (m)
      5: East  offset (m)
      6: Up    offset (m)
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
            tow = float(parts[1]); week = int(float(parts[2]))
            north_m = float(parts[3]); east_m = float(parts[4]); up_m = float(parts[5])
            roll = float(parts[6]); pitch = float(parts[7]); yaw = float(parts[8])
        except Exception:
            continue
        recs.append((img, week, tow, north_m, east_m, up_m, roll, pitch, yaw))
    return pd.DataFrame(
        recs,
        columns=["image", "gps_week", "gps_tow_s", "North_m", "East_m", "Up_m",
                 "Roll_deg", "Pitch_deg", "Yaw_deg"]
    )

# ------------------------------
# UI
# ------------------------------
st.title("Jamie D PPK Processor")
st.caption("PPK (RTKLIB rnx2rtkp) → time match → ENU→ECEF per-event offsets → EXIF CSV (Yaw, Pitch, Roll)")

with st.expander("Offset convention & timing", expanded=False):
    st.markdown(
        "- **Offsets file (meters):** `Image, TOW(s), GPS week, North(m), East(m), Up(m), Roll, Pitch, Yaw`  \n"
        "- If offsets describe **APC relative to Camera** (antenna is N/E/U of camera), "
        "leave the box checked (default). We compute **Camera = APC − [N,E,U]**.  \n"
        "- If offsets describe **Camera relative to APC**, uncheck it and we compute **Camera = APC + [N,E,U]**.  \n"
        "- **Exposure delay:** If GNSS timestamp is earlier than exposure, use a positive delay."
    )

subtract_mode = st.checkbox(
    "Offsets describe **APC relative to Camera** (subtract: Camera = APC − [N,E,U])",
    value=True
)
exp_delay = st.number_input(
    "Global exposure delay (seconds, applied to event time before matching)",
    value=0.062, step=0.001, format="%.3f",
    help="Positive = shutter fires AFTER logged time. Example: 0.062 means +62 ms."
)
show_diag = st.checkbox("Show diagnostics in app (won’t be included in CSV)", value=True)

colL, colR = st.columns(2)
with colL:
    st.subheader("Rover OBS")
    rover_obs_up = st.file_uploader("Rover RINEX OBS", type=None,
                                    help="Any extension is fine; passed through to rnx2rtkp.")
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
    st.subheader("Events (TXT/CSV, meters)")
    st.caption("No headers. Columns: Image, TOW(s), GPS week, North(m), East(m), Up(m), Roll(deg), Pitch(deg), Yaw(deg).")
    events_up = st.file_uploader("Events file", type=["txt", "csv"])
    if events_up: st.write(f"**Loaded:** `{events_up.name}`")

st.markdown("---")
run_clicked = st.button("Run PPK Processing", use_container_width=True)

# ------------------------------
# State
# ------------------------------
if "ppk_out_path" not in st.session_state: st.session_state["ppk_out_path"] = ""
if "ppk_cmd" not in st.session_state:      st.session_state["ppk_cmd"] = ""
if "ppk_out_log" not in st.session_state:  st.session_state["ppk_out_log"] = ""
if "pos_df" not in st.session_state:       st.session_state["pos_df"] = pd.DataFrame()
if "events_df" not in st.session_state:    st.session_state["events_df"] = pd.DataFrame()

# ------------------------------
# Core
# ------------------------------
def match_events_to_pos(pos_df: pd.DataFrame,
                        events_df: pd.DataFrame,
                        tol_s: float = 2.0,
                        subtract_mode: bool = True,
                        delay_s: float = 0.0,
                        diagnostics: bool = False) -> pd.DataFrame:
    """
    For each event (week,tow), find nearest pos epoch with same week and |dt|<=tol_s.
    Convert APC -> Camera using ENU/ECEF with per-event offsets and optional exposure delay.
    Returns full DataFrame with metadata columns for on-screen diagnostics;
    these extra columns are removed for the downloadable CSV.
    """
    if pos_df.empty or events_df.empty:
        cols = ["Img","Lat","Long","Alt","Yaw","Pitch","Roll","X acc","Y acc","Z acc"]
        if diagnostics: cols += ["Δt_s","mode","delay_s","dE_applied_m","dN_applied_m","dU_applied_m",
                                 "E_recovered_m","N_recovered_m","U_recovered_m"]
        return pd.DataFrame(columns=cols)

    out_rows = []
    for wk in sorted(events_df["gps_week"].unique()):
        pos_w = pos_df[pos_df["gps_week"] == wk]
        if pos_w.empty: continue

        tow_pos = pos_w["gps_tow_s"].to_numpy()
        lat_pos = pos_w["lat_deg"].to_numpy()
        lon_pos = pos_w["lon_deg"].to_numpy()
        hgt_pos = pos_w["hgt_m"].to_numpy()

        events_w = events_df[events_df["gps_week"] == wk]
        for _, ev in events_w.iterrows():
            tow_event = float(ev["gps_tow_s"]) + float(delay_s)
            idx = np.searchsorted(tow_pos, tow_event)
            cand = []
            if 0 <= idx < len(tow_pos): cand.append(idx)
            if idx - 1 >= 0:            cand.append(idx - 1)

            best = None; best_dt = None
            for ci in cand:
                dt = abs(tow_pos[ci] - tow_event)
                if best_dt is None or dt < best_dt: best_dt, best = dt, ci
            if best is None or (best_dt is not None and best_dt > tol_s):
                continue

            base_lat = float(lat_pos[best]); base_lon = float(lon_pos[best]); base_h = float(hgt_pos[best])
            delta_t = float(tow_pos[best] - tow_event)

            n = float(ev["North_m"]); e = float(ev["East_m"]); u = float(ev["Up_m"])
            if subtract_mode:
                dN, dE, dU = -n, -e, -u
                mode_label = "subtract (APC rel. Camera)"
            else:
                dN, dE, dU = +n, +e, +u
                mode_label = "add (Camera rel. APC)"

            (adj_lat, adj_lon, adj_h), (Xa,Ya,Za), (Xc,Yc,Zc) = apply_offsets_via_ecef(base_lat, base_lon, base_h, dN, dE, dU)

            row = [
                ev["image"], adj_lat, adj_lon, adj_h,
                ev["Yaw_deg"], ev["Pitch_deg"], ev["Roll_deg"],
                0.02, 0.02, 0.03
            ]
            if diagnostics:
                # Recover ENU delta from ECEF (should match dE/dN/dU used)
                Erec, Nrec, Urec = ecef_delta_to_enu(base_lat, base_lon, Xa,Ya,Za, Xc,Yc,Zc)
                row += [delta_t, mode_label, float(delay_s), dE, dN, dU, Erec, Nrec, Urec]
            out_rows.append(row)

    cols = ["Img","Lat","Long","Alt","Yaw","Pitch","Roll","X acc","Y acc","Z acc"]
    if diagnostics:
        cols += ["Δt_s","mode","delay_s","dE_applied_m","dN_applied_m","dU_applied_m",
                 "E_recovered_m","N_recovered_m","U_recovered_m"]
    return pd.DataFrame(out_rows, columns=cols)

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

        with st.expander("Command", expanded=False): st.code(cmd)
        with st.expander("RTKLIB stdout / stderr", expanded=False): st.code(out or "(no stdout)")

        if os.path.exists(out_pos) and os.path.getsize(out_pos) > 0:
            st.success(f"PPK solve complete. **.pos generated** (epochs parsed ≈ {n_epochs}).")
            st.session_state["ppk_out_path"] = out_pos
            with open(out_pos, "rb") as f:
                st.download_button("Download solution.pos", f.read(), "solution.pos", "text/plain")
            st.session_state["pos_df"] = parse_rtklib_pos(out_pos)
            st.info(f"Parsed {len(st.session_state['pos_df'])} epochs from .pos.")
        else:
            st.error("PPK failed or .pos not created. Check the RTKLIB log above.")
            st.session_state["ppk_out_path"] = ""; st.session_state["pos_df"] = pd.DataFrame()

        if events_path and os.path.exists(events_path):
            st.session_state["events_df"] = parse_events_no_headers(events_path)
            st.info(f"Events parsed: **{len(st.session_state['events_df'])}** rows.")
        else:
            st.session_state["events_df"] = pd.DataFrame()
            st.warning("Upload an events file to produce the EXIF CSV.")

# ------------------------------
# Build CSV
# ------------------------------
if not st.session_state["pos_df"].empty and not st.session_state["events_df"].empty:
    st.markdown("---")
    st.subheader("Build EXIF CSV (ENU→ECEF per-event offsets; exposure delay applied)")

    tol = st.slider("Time matching tolerance (seconds)", 0.1, 5.0, 2.0, 0.1,
                    help="Max allowed |TOW(event_corrected)-TOW(.pos)| for same GPS week.")
    out_df = match_events_to_pos(
        st.session_state["pos_df"], st.session_state["events_df"],
        tol_s=tol, subtract_mode=subtract_mode, delay_s=exp_delay,
        diagnostics=show_diag
    )

    if out_df.empty:
        st.warning("No matches within tolerance. Try increasing the tolerance.")
    else:
        # Show diagnostics table (app only)
        if show_diag:
            st.success(f"Matched **{len(out_df)}** images. (Diagnostics shown below; not included in CSV.)")
            st.dataframe(out_df.head(20), use_container_width=True)
        else:
            st.success(f"Matched **{len(out_df)}** images.")

        # Only include the 10 standard EXIF columns in the downloadable CSV
        csv_cols = ["Img","Lat","Long","Alt","Yaw","Pitch","Roll","X acc","Y acc","Z acc"]
        csv_bytes = out_df[csv_cols].to_csv(index=False, float_format="%.7f").encode("utf-8")
        st.download_button("Download EXIF CSV", csv_bytes, "exif_ppk.csv", "text/csv")
else:
    if st.session_state["pos_df"].empty:
        st.info("Run PPK first to create a .pos.")
    elif st.session_state["events_df"].empty:
        st.info("Upload an events file to produce the EXIF CSV.")

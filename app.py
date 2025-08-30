import os
import math
import tempfile
import subprocess
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import shutil
import pathlib

# ------------------------------
# Page setup + style
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
            lat = lat_new
            break
        lat = lat_new
    sL = math.sin(lat)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sL * sL)
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

def add_vec(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def scale_vec(v, s):
    return (v[0]*s, v[1]*s, v[2]*s)

def apply_offsets_via_ecef(lat_deg: float, lon_deg: float, h_m: float,
                           dN_m: float, dE_m: float, dU_m: float):
    """
    Apply ENU vector [dE, dN, dU] (meters) at APC to get camera center using ECEF math.
    Returns (lat, lon, h), (Xapc,Yapc,Zapc), (Xcam,Ycam,Zcam)
    """
    X, Y, Z = geodetic_to_ecef(lat_deg, lon_deg, h_m)
    e, n, u = enu_basis(lat_deg, lon_deg)
    d_ecef = add_vec(add_vec(scale_vec(e, dE_m), scale_vec(n, dN_m)), scale_vec(u, dU_m))
    Xc, Yc, Zc = add_vec((X, Y, Z), d_ecef)
    return ecef_to_geodetic(Xc, Yc, Zc), (X, Y, Z), (Xc, Yc, Zc)

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
    """
    Parse RTKLIB .pos; extract GPS week, TOW, lat, lon, hgt.
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
# ------------------------------
# UI
# ------------------------------
st.title("Jamie D PPK Processor")
st.caption("PPK (RTKLIB rnx2rtkp) → time interpolation in ECEF → ENU offsets (per-event) → EXIF CSV")

with st.expander("Offset convention & timing", expanded=False):
    st.markdown(
        "- **Offsets file (meters):** `Image, TOW(s), GPS week, North(m), East(m), Up(m), Roll, Pitch, Yaw`  \n"
        "- If offsets describe **APC relative to Camera** (antenna is N/E/U of camera), "
        "leave the box checked (default). We compute **Camera = APC − [N,E,U]**.  \n"
        "- If offsets describe **Camera relative to APC**, uncheck it and we compute **Camera = APC + [N,E,U]**.  \n"
        "- **Exposure delay:** GNSS earlier than exposure → use positive delay (e.g., +0.062 s)."
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
interp_on = st.checkbox("Interpolate between .pos epochs in time (ECEF)", value=True)
show_diag = st.checkbox("Show diagnostics in app (not included in CSV)", value=False)

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
# Matching with ECEF interpolation
# ------------------------------
def interp_ecef_at_time(tow_pos, lat_pos, lon_pos, hgt_pos, t_event, tol_s, do_interp=True):
    """
    Return (lat,lon,h) at time t_event by either:
      - linear interpolation between surrounding epochs (ECEF), or
      - nearest neighbor if interpolation not possible or disabled.
    Skips if gap > tol_s on both sides.
    """
    n = len(tow_pos)
    if n == 0:
        return None

    idx = np.searchsorted(tow_pos, t_event)

    # Helper: convert one epoch to ECEF
    def ecef_at(i):
        return geodetic_to_ecef(float(lat_pos[i]), float(lon_pos[i]), float(hgt_pos[i]))

    # Try interpolation if enabled and we have both sides
    if do_interp and 0 < idx < n:
        t0, t1 = float(tow_pos[idx-1]), float(tow_pos[idx])
        if (t_event >= t0) and (t_event <= t1) and (t1 - t0) > 0:
            if (abs(t_event - t0) <= tol_s) and (abs(t1 - t_event) <= tol_s):
                X0, Y0, Z0 = ecef_at(idx-1)
                X1, Y1, Z1 = ecef_at(idx)
                a = (t_event - t0) / (t1 - t0)
                Xi = (1-a)*X0 + a*X1
                Yi = (1-a)*Y0 + a*Y1
                Zi = (1-a)*Z0 + a*Z1
                return ecef_to_geodetic(Xi, Yi, Zi)

    # Fall back: nearest neighbor within tol
    cand = []
    if 0 <= idx < n:          cand.append(idx)
    if idx - 1 >= 0:          cand.append(idx - 1)
    if not cand:              return None

    best = None; best_dt = None
    for ci in cand:
        dt = abs(float(tow_pos[ci]) - t_event)
        if best_dt is None or dt < best_dt:
            best_dt, best = dt, ci
    if best_dt is None or best_dt > tol_s:
        return None
    return float(lat_pos[best]), float(lon_pos[best]), float(hgt_pos[best])

def match_events_to_pos(pos_df: pd.DataFrame,
                        events_df: pd.DataFrame,
                        tol_s: float = 2.0,
                        subtract_mode: bool = True,
                        delay_s: float = 0.0,
                        do_interp: bool = True,
                        diagnostics: bool = False) -> pd.DataFrame:
    """
    For each event (week,tow), compute base position at corrected event time using ECEF interpolation,
    then apply per-event N/E/U offsets (subtract or add).
    Returns DF for UI (diagnostics optional). CSV download will include only EXIF columns.
    """
    cols = ["Img","Lat","Long","Alt","Yaw","Pitch","Roll","X acc","Y acc","Z acc"]
    if diagnostics:
        cols += ["Δt_to_nearest_s","mode","delay_s"]
    out = []

    if pos_df.empty or events_df.empty:
        return pd.DataFrame(columns=cols)

    for wk in sorted(events_df["gps_week"].unique()):
        pos_w = pos_df[pos_df["gps_week"] == wk]
        if pos_w.empty: continue

        tow_pos = pos_w["gps_tow_s"].to_numpy()
        lat_pos = pos_w["lat_deg"].to_numpy()
        lon_pos = pos_w["lon_deg"].to_numpy()
        hgt_pos = pos_w["hgt_m"].to_numpy()

        events_w = events_df[events_df["gps_week"] == wk]
        for _, ev in events_w.iterrows():
            t_ev = float(ev["gps_tow_s"]) + float(delay_s)

            # For diagnostics only: nearest Δt (before interpolation)
            idx = np.searchsorted(tow_pos, t_ev)
            dtn = None
            if 0 <= idx < len(tow_pos):
                dtn = abs(float(tow_pos[idx]) - t_ev)
            if idx-1 >= 0:
                dt2 = abs(float(tow_pos[idx-1]) - t_ev)
                dtn = min(dtn, dt2) if dtn is not None else dt2

            base = interp_ecef_at_time(tow_pos, lat_pos, lon_pos, hgt_pos, t_ev, tol_s, do_interp)
            if base is None:
                continue
            base_lat, base_lon, base_h = base

            n = float(ev["North_m"]); e = float(ev["East_m"]); u = float(ev["Up_m"])
            if subtract_mode:
                dN, dE, dU = -n, -e, -u
                mode_label = "subtract (APC rel. Camera)"
            else:
                dN, dE, dU = +n, +e, +u
                mode_label = "add (Camera rel. APC)"

            (adj_lat, adj_lon, adj_h), _, _ = apply_offsets_via_ecef(base_lat, base_lon, base_h, dN, dE, dU)

            row = [ev["image"], adj_lat, adj_lon, adj_h,
                   ev["Yaw_deg"], ev["Pitch_deg"], ev["Roll_deg"],
                   0.02, 0.02, 0.03]
            if diagnostics:
                row += [float(dtn) if dtn is not None else None, mode_label, float(delay_s)]
            out.append(row)

    return pd.DataFrame(out, columns=cols)

# ------------------------------
# Run PPK
# ------------------------------
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
    st.subheader("Build EXIF CSV (ECEF time interpolation + per-event ENU offsets)")
    tol = st.slider("Time matching tolerance (seconds)", 0.1, 5.0, 2.0, 0.1,
                    help="Max allowed temporal distance to surrounding epochs for interpolation/nearest.")
    out_df = match_events_to_pos(
        st.session_state["pos_df"], st.session_state["events_df"],
        tol_s=tol, subtract_mode=subtract_mode,
        delay_s=exp_delay, do_interp=interp_on, diagnostics=show_diag
    )

    if out_df.empty:
        st.warning("No matches within tolerance. Try increasing the tolerance.")
    else:
        if show_diag:
            st.info("Diagnostics columns are visible below but are NOT included in the CSV download.")
            st.dataframe(out_df.head(20), use_container_width=True)
        else:
            st.dataframe(out_df.head(20)[["Img","Lat","Long","Alt","Yaw","Pitch","Roll","X acc","Y acc","Z acc"]],
                         use_container_width=True)

        # Only standard EXIF columns in the CSV
        csv_cols = ["Img","Lat","Long","Alt","Yaw","Pitch","Roll","X acc","Y acc","Z acc"]
        csv_bytes = out_df[csv_cols].to_csv(index=False, float_format="%.7f").encode("utf-8")
        st.download_button("Download EXIF CSV", csv_bytes, "exif_ppk.csv", "text/csv")
else:
    if st.session_state["pos_df"].empty:
        st.info("Run PPK first to create a .pos.")
    elif st.session_state["events_df"].empty:
        st.info("Upload an events file to produce the EXIF CSV.")
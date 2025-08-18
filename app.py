# app.py
import os
import io
import math
import json
import hashlib
import shutil
import tempfile
import subprocess
from datetime import datetime, timezone
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------
# Page setup + BUTTON COLOR CSS
# ------------------------------
st.set_page_config(page_title="Jamie D PPK Processor", layout="wide")

# Make the main action button green
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
# RTKLIB bootstrap (prefer bundled binary)
# ------------------------------
def prepend_bin_to_path():
    """Prepend ./bin and /app/bin to PATH so rnx2rtkp is found (local + Streamlit Cloud)."""
    bin_dirs = [
        os.path.join(os.getcwd(), "bin"),  # repo-local
        "/app/bin",                        # Streamlit Cloud convention
    ]
    for d in bin_dirs:
        if os.path.isdir(d):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

def which_rnx2rtkp() -> Tuple[bool, str]:
    """Return (available, message/version/help-first-line). Some builds use -? for help."""
    try:
        p = subprocess.run(["rnx2rtkp", "-?"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        first = (p.stdout or "").splitlines()[0] if p.stdout else "rnx2rtkp available"
        return True, first
    except FileNotFoundError:
        return False, "rnx2rtkp not found on PATH"

def interpret_rtk_output(out: str) -> list:
    """Human-friendly troubleshooting tips based on RTKLIB stdout/stderr."""
    if not out:
        return []
    t = out.lower()
    tips = []
    if "no observation data" in t or "no obs data" in t:
        tips.append("No observation data detected. Verify Rover/Base OBS files and RINEX versions.")
    if "no nav data" in t or "nav data error" in t:
        tips.append("Navigation data issue. Ensure Base NAV matches the OBS time range (correct day/week).")
    if "mismatch" in t and "time" in t:
        tips.append("Time mismatch between datasets. Check GPS week/TOW and time zones.")
    if "solution status" in t and "single" in t:
        tips.append("Solution is SINGLE. Consider better base data, longer overlap, or a tuned .conf.")
    if "ant" in t and "offset" in t:
        tips.append("Antenna offset warning. Confirm lever arms and reference point conventions.")
    return tips

def sha1(path: str) -> str:
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "NA"

prepend_bin_to_path()
_rnx_ok, _rnx_msg = which_rnx2rtkp()

# Sidebar status
st.sidebar.header("RTKLIB Status")
st.sidebar.write(_rnx_msg if _rnx_ok else "rnx2rtkp not available (bundle bin/rnx2rtkp)")
if not _rnx_ok:
    st.sidebar.warning("Bundle a Linux rnx2rtkp at bin/rnx2rtkp (executable) for Streamlit Cloud.")

# ------------------------------
# Core Helpers
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

def run_rnx2rtkp(rover_obs: str, base_obs: str, base_nav: str, out_pos_path: str, conf_path: Optional[str] = None) -> Tuple[str, str, int]:
    """
    Run RTKLIB rnx2rtkp to produce a .pos.
    Returns (cmd, combined_output, parsed_epochs_count_guess)
    """
    cmd = ["rnx2rtkp", rover_obs, base_obs, base_nav, "-o", out_pos_path]
    if conf_path:
        cmd.extend(["-k", conf_path])
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    except FileNotFoundError:
        out = "ERROR: rnx2rtkp not found on PATH. Ensure ./bin/rnx2rtkp (bundled) or install RTKLIB."
        return (" ".join(cmd), out, 0)

    # best-effort epoch count
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
            lat_idx, lon_idx, hgt_idx = None, None, None
            for i in range(len(parts)):
                try:
                    v = float(parts[i])
                except:
                    continue
                if -90 <= v <= 90 and lat_idx is None:
                    if i + 1 < len(parts):
                        try:
                            v2 = float(parts[i + 1])
                            if -180 <= v2 <= 180:
                                lat_idx, lon_idx = i, i + 1
                                if i + 2 < len(parts):
                                    try:
                                        _ = float(parts[i + 2])
                                        hgt_idx = i + 2
                                    except:
                                        pass
                                break
                        except:
                            pass
            week, tow = None, None
            for i in range(min(4, len(parts) - 1)):
                try:
                    w = int(float(parts[i])); t = float(parts[i + 1])
                    if 800 <= w <= 4000 and 0 <= t < 700000:
                        week, tow = w, t
                        break
                except:
                    continue
            if lat_idx is not None and lon_idx is not None and hgt_idx is not None and week is not None and tow is not None:
                try:
                    lat = float(parts[lat_idx]); lon = float(parts[lon_idx]); hgt = float(parts[hgt_idx])
                except:
                    continue
                rows.append((week, tow, lat, lon, hgt))
    return pd.DataFrame(rows, columns=["gps_week", "gps_tow_s", "lat_deg", "lon_deg", "hgt_m"])

def parse_events_no_headers(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        return pd.DataFrame()
    with open(file_path, "r", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    records = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 9:
            parts = [p.strip() for p in ln.replace("\t", ",").split(",")]
        if len(parts) < 9:
            continue
        img = parts[0]
        try:
            tow = float(parts[1]); week = int(float(parts[2]))
            n = float(parts[3]); e = float(parts[4]); u = float(parts[5])
            roll = float(parts[6]); pitch = float(parts[7]); yaw = float(parts[8])
        except:
            continue
        records.append((img, week, tow, n, e, u, roll, pitch, yaw))
    return pd.DataFrame(records, columns=[
        "image","gps_week","gps_tow_s","N_m","E_m","U_m","Roll_deg","Pitch_deg","Yaw_deg"
    ])

def apply_neu_offsets(lat_deg: float, lon_deg: float, h_m: float,
                      n_m: float, e_m: float, u_m: float) -> Tuple[float, float, float]:
    R = 6378137.0
    lat_rad = math.radians(lat_deg)
    dlat = (n_m / R) * (180.0 / math.pi)
    dlon = (e_m / (R * math.cos(lat_rad))) * (180.0 / math.pi)
    dalt = u_m
    return lat_deg + dlat, lon_deg + dlon, h_m + dalt

def match_events_to_pos(pos_df: pd.DataFrame, events_df: pd.DataFrame, tol_s: float = 2.0) -> pd.DataFrame:
    if pos_df.empty or events_df.empty:
        return pd.DataFrame(columns=["Img","Lat","Long","Alt","Yaw","Pitch","Roll","X acc","Y acc","Z acc"])
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
            cand_idxs = []
            if 0 <= idx < len(tow_pos): cand_idxs.append(idx)
            if idx - 1 >= 0: cand_idxs.append(idx - 1)
            best, best_dt = None, None
            for ci in cand_idxs:
                dt = abs(tow_pos[ci] - tow)
                if best_dt is None or dt < best_dt:
                    best_dt, best = dt, ci
            if best is None or (best_dt is not None and best_dt > tol_s):
                continue
            base_lat, base_lon, base_h = lat_pos[best], lon_pos[best], hgt_pos[best]
            adj_lat, adj_lon, adj_h = apply_neu_offsets(base_lat, base_lon, base_h, ev["N_m"], ev["E_m"], ev["U_m"])
            out_rows.append([
                ev["image"], adj_lat, adj_lon, adj_h,
                ev["Yaw_deg"], ev["Pitch_deg"], ev["Roll_deg"],
                0.02, 0.02, 0.03
            ])
    return pd.DataFrame(out_rows, columns=["Img","Lat","Long","Alt","Yaw","Pitch","Roll","X acc","Y acc","Z acc"])

# ------------------------------
# Validation & classification (UI/UX polish)
# ------------------------------
def validate_events(df: pd.DataFrame) -> list:
    errs = []
    req = ["image","gps_week","gps_tow_s","N_m","E_m","U_m","Roll_deg","Pitch_deg","Yaw_deg"]
    for c in req:
        if c not in df.columns:
            errs.append(f"Missing column {c}")
    if not df.empty:
        if not df["gps_week"].between(800, 4000).all():
            errs.append("GPS week out of expected range (800–4000).")
        if not df["gps_tow_s"].between(0, 604800).all():
            errs.append("GPS TOW out of expected range (0–604800).")
    return errs

def classify_rinex(path: str) -> str:
    """Return 'OBS' | 'NAV' | 'UNKNOWN' based on header/extension."""
    try:
        with open(path, "r", errors="ignore") as f:
            head = "".join([next(f) for _ in range(40)])
        if "RINEX VERSION" in head and "NAVIGATION" in head:
            return "NAV"
        if "RINEX VERSION" in head and ("OBSERVATION DATA" in head or "OBSERVATION" in head):
            return "OBS"
    except Exception:
        pass
    ext = os.path.splitext(path.lower())[1]
    if ext in [".nav", ".n", ".25n", ".rnx", ".gnav"]: return "NAV"
    if ext in [".obs", ".o", ".25o"]: return "OBS"
    return "UNKNOWN"

# ------------------------------
# UI
# ------------------------------
st.title("Jamie D PPK Processor")
st.caption("PPK solve (RTKLIB rnx2rtkp) → event matching + NEU camera offsets → EXIF CSV (Yaw, Pitch, Roll)")

with st.sidebar:
    st.markdown("### Controls")
    reset = st.button("Reset session state")
    if reset:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

colL, colR = st.columns(2)

with colL:
    st.subheader("Rover OBS")
    rover_obs_up = st.file_uploader(
        "Drag & drop Rover RINEX OBS (any extension is OK, e.g., .obs, .25o, etc.)",
        type=None,
        help="Any extension is fine; passed directly to rnx2rtkp."
    )
    if rover_obs_up: st.write(f"**Loaded:** `{rover_obs_up.name}`")

with colR:
    st.subheader("Base OBS")
    base_obs_up = st.file_uploader(
        "Drag & drop Base RINEX OBS (any extension OK, e.g., .obs, .25o, etc.)",
        type=None
    )
    if base_obs_up: st.write(f"**Loaded:** `{base_obs_up.name}`")

with colL:
    st.subheader("Base NAV")
    base_nav_up = st.file_uploader(
        "Drag & drop Base RINEX NAV (any extension OK, e.g., .nav, .25n, etc.)",
        type=None
    )
    if base_nav_up: st.write(f"**Loaded:** `{base_nav_up.name}`")

with colR:
    st.subheader("Events (optional, TXT/CSV)")
    st.caption("Rows: Image, TOW(s), GPS week, N(m), E(m), U(m), Roll(deg), Pitch(deg), Yaw(deg); cols 10–12 ignored")
    events_up = st.file_uploader("Drag & drop events file (no headers required)", type=["txt", "csv"])
    if events_up: st.write(f"**Loaded:** `{events_up.name}`")

st.markdown("—")

# Optional RTKLIB config
conf_up = st.file_uploader("Optional: RTKLIB config (.conf) to fine-tune processing", type=["conf"])

run_clicked = st.button("Run PPK Processing", use_container_width=True)

# Scratch area
for key, default in [
    ("ppk_out_path",""), ("ppk_cmd",""), ("ppk_out_log",""),
    ("pos_df", pd.DataFrame()), ("events_df", pd.DataFrame()),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Persist last-used tolerance
if "tol_seconds" not in st.session_state:
    st.session_state["tol_seconds"] = 2.0

if run_clicked:
    if not _rnx_ok:
        st.error("RTKLIB `rnx2rtkp` is not available. Bundle `bin/rnx2rtkp` in the repo or install RTKLIB, then reload.")
    else:
        with st.status("Processing PPK…", expanded=True) as status:
            st.write("Saving uploads…")
            rover_path   = save_upload_to_tmp(rover_obs_up) if rover_obs_up else ""
            base_obs_path= save_upload_to_tmp(base_obs_up) if base_obs_up else ""
            base_nav_path= save_upload_to_tmp(base_nav_up) if base_nav_up else ""
            events_path  = save_upload_to_tmp(events_up) if events_up else ""
            conf_path    = save_upload_to_tmp(conf_up, suffix=".conf") if conf_up else None

            if not rover_path or not base_obs_path or not base_nav_path:
                st.error("Provide Rover OBS, Base OBS, and Base NAV files.")
                status.update(label="PPK aborted — missing inputs", state="error", expanded=False)
            else:
                # Light classification to detect obvious mistakes
                st.write("Checking file types…")
                kinds = {
                    "Rover?": (rover_path, classify_rinex(rover_path)),
                    "Base OBS?": (base_obs_path, classify_rinex(base_obs_path)),
                    "Base NAV?": (base_nav_path, classify_rinex(base_nav_path)),
                }
                bad = []
                if kinds["Rover?"][1] != "OBS": bad.append("Rover file does not look like OBS.")
                if kinds["Base OBS?"][1] != "OBS": bad.append("Base file does not look like OBS.")
                if kinds["Base NAV?"][1] != "NAV": bad.append("NAV file does not look like NAV.")
                if bad:
                    st.warning(" | ".join(bad))

                st.write("Running RTKLIB rnx2rtkp…")
                out_pos = os.path.join(tempfile.gettempdir(), "solution.pos")
                cmd, out, n_epochs = run_rnx2rtkp(rover_path, base_obs_path, base_nav_path, out_pos, conf_path=conf_path)
                st.session_state["ppk_cmd"] = cmd
                st.session_state["ppk_out_log"] = out

                with st.expander("Command", expanded=False):
                    st.code(cmd)
                with st.expander("RTKLIB stdout / stderr", expanded=False):
                    st.code(out or "(no stdout)")

                tips = interpret_rtk_output(out)
                if tips:
                    st.info("Troubleshooting tips:\n- " + "\n- ".join(tips))

                if os.path.exists(out_pos) and os.path.getsize(out_pos) > 0:
                    st.write("Parsing .pos…")
                    pos_df = parse_rtklib_pos(out_pos)
                    st.session_state["pos_df"] = pos_df
                    st.session_state["ppk_out_path"] = out_pos
                    st.success(f"PPK solve complete. **.pos generated** — epochs parsed ≈ {n_epochs}.")
                    with open(out_pos, "rb") as f:
                        st.download_button("Download solution.pos", data=f.read(), file_name="solution.pos", mime="text/plain")
                    st.info(f"Parsed {len(pos_df)} epochs from .pos.")
                else:
                    st.error("PPK failed or .pos not created. Check the RTKLIB log above.")
                    st.session_state["ppk_out_path"] = ""
                    st.session_state["pos_df"] = pd.DataFrame()

                if events_path and os.path.exists(events_path):
                    st.write("Parsing events…")
                    events_df = parse_events_no_headers(events_path)
                    errs = validate_events(events_df)
                    if errs:
                        st.warning("Events validation:\n- " + "\n- ".join(errs))
                    st.session_state["events_df"] = events_df
                    st.info(f"Events parsed: **{len(events_df)}** rows.")
                else:
                    st.session_state["events_df"] = pd.DataFrame()
                    st.warning("Upload an events file to produce the EXIF CSV.")

                # Reproducibility metadata
                st.write("Preparing run metadata…")
                run_meta = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "rnx2rtkp_version": _rnx_msg if _rnx_ok else "NA",
                    "cmd": st.session_state["ppk_cmd"],
                    "epochs_parsed": int(n_epochs),
                    "files_sha1": {
                        "rover": sha1(rover_path) if rover_path else "NA",
                        "base_obs": sha1(base_obs_path) if base_obs_path else "NA",
                        "base_nav": sha1(base_nav_path) if base_nav_path else "NA",
                        "conf": sha1(conf_path) if conf_path else "NA",
                        "solution_pos": sha1(st.session_state["ppk_out_path"]) if st.session_state["ppk_out_path"] else "NA"
                    }
                }
                st.download_button("Download run_metadata.json",
                                   data=json.dumps(run_meta, indent=2).encode("utf-8"),
                                   file_name="run_metadata.json")

                status.update(label="PPK processing complete", state="complete", expanded=False)

# If we have both, allow building the EXIF CSV
pos_ready = not st.session_state["pos_df"].empty
events_ready = not st.session_state["events_df"].empty

if pos_ready and events_ready:
    st.markdown("---")
    st.subheader("Build EXIF CSV from matched PPK + Events (includes NEU offsets; outputs Yaw, Pitch, Roll)")

    tol = st.slider("Time matching tolerance (seconds)",
                    0.1, 5.0, float(st.session_state.get("tol_seconds", 2.0)), 0.1,
                    help="Max allowed time difference between event TOW and .pos TOW for the same GPS week.")
    st.session_state["tol_seconds"] = tol

    out_df = match_events_to_pos(st.session_state["pos_df"], st.session_state["events_df"], tol_s=tol)

    if out_df.empty:
        st.warning("No matches within tolerance. Try increasing the tolerance.")
    else:
        st.success(f"Matched **{len(out_df)}** images.")
        st.dataframe(out_df.head(20), use_container_width=True)

        # Enforce 7+ decimal places and locale guards
        csv_bytes = out_df.to_csv(
            index=False,
            float_format="%.7f",
            sep=",",
            decimal=".",
            na_rep="",
            lineterminator="\n"
        ).encode("utf-8")

        # Better default filename
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download EXIF CSV",
            data=csv_bytes,
            file_name=f"exif_ppk_{stamp}.csv",
            mime="text/csv"
        )
else:
    if st.session_state["pos_df"].empty:
        st.info("Run PPK first to create a .pos.")
    elif st.session_state["events_df"].empty:
        st.info("Upload an events file to produce the EXIF CSV.")

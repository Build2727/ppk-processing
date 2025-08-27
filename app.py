# app.py
import os
import io
import sys
import math
import json
import time
import shutil
import pathlib
import tempfile
import subprocess
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st

# =========================
# 1) Make sure RTKLIB exists (build on first run)
# =========================
def ensure_rnx2rtkp() -> str:
    """
    Ensure a glibc-compatible rnx2rtkp is available on this host.
    If not present in cache, build RTKLIB's rnx2rtkp from source.
    Returns the path to the binary and prepends its dir to PATH.
    """
    cache_dir = pathlib.Path.home() / ".cache" / "ppk" / "bin"
    rnx_path = cache_dir / "rnx2rtkp"

    # Put cache bin first on PATH
    os.environ["PATH"] = f"{str(cache_dir)}{os.pathsep}{os.environ.get('PATH','')}"

    if rnx_path.exists():
        return str(rnx_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    st.sidebar.info("Building RTKLIB (first run)… one-time step")

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = pathlib.Path(tmpd)
        # Clone RTKLIB
        subprocess.run(
            ["git", "clone", "https://github.com/tomojitakasu/RTKLIB.git", str(tmp / "RTKLIB")],
            check=True,
        )
        # Build rnx2rtkp
        mk_dir = tmp / "RTKLIB" / "app" / "rnx2rtkp" / "gcc"
        subprocess.run(["make", "-C", str(mk_dir), "-j"], check=True)
        built = mk_dir / "rnx2rtkp"
        if not built.exists():
            raise RuntimeError("Build OK but rnx2rtkp not found")
        shutil.copy2(str(built), str(rnx_path))
        os.chmod(str(rnx_path), 0o755)

    st.sidebar.success("RTKLIB is ready.")
    return str(rnx_path)

RNX2RTKP = ensure_rnx2rtkp()

# =========================
# 2) Geodesy helpers (WGS84)
# =========================
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)

def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    a = WGS84_A
    e2 = WGS84_E2
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    X = (N + h_m) * cos_lat * math.cos(lon)
    Y = (N + h_m) * cos_lat * math.sin(lon)
    Z = (N * (1 - e2) + h_m) * sin_lat
    return X, Y, Z

def ecef_to_geodetic(X, Y, Z):
    # Bowring’s method
    a = WGS84_A
    e2 = WGS84_E2
    b = a * (1 - WGS84_F)
    ep2 = (a*a - b*b) / (b*b)

    p = math.sqrt(X*X + Y*Y)
    th = math.atan2(a * Z, b * p)
    lon = math.atan2(Y, X)
    lat = math.atan2(Z + ep2 * b * math.sin(th)**3,
                     p - e2 * a * math.cos(th)**3)
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - N
    return math.degrees(lat), math.degrees(lon), h

def enu_axes(lat_deg, lon_deg):
    """Return unit E,N,U axes in ECEF at a given geodetic lat/lon."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    # ECEF unit vectors for local ENU
    e = np.array([-math.sin(lon), math.cos(lon), 0.0])
    n = np.array([-math.sin(lat)*math.cos(lon),
                  -math.sin(lat)*math.sin(lon),
                   math.cos(lat)])
    u = np.array([ math.cos(lat)*math.cos(lon),
                   math.cos(lat)*math.sin(lon),
                   math.sin(lat)])
    return e, n, u

def apply_enu_offset(lat_deg, lon_deg, h_m, dN, dE, dU):
    """Apply (N,E,U) offsets in meters by converting to ECEF, adding vector, converting back."""
    X, Y, Z = geodetic_to_ecef(lat_deg, lon_deg, h_m)
    e, n, u = enu_axes(lat_deg, lon_deg)
    d_ecef = dE * e + dN * n + dU * u
    Xp, Yp, Zp = X + d_ecef[0], Y + d_ecef[1], Z + d_ecef[2]
    return ecef_to_geodetic(Xp, Yp, Zp)

# =========================
# 3) rnx2rtkp runner (optional)
# =========================
def run_rnx2rtkp(rover_obs_path, base_obs_path=None, nav_path=None) -> (str, str):
    """
    Run rnx2rtkp to produce a .pos. Returns (stdout, pos_path).
    The command here is minimal; extend if you need SP3, config, etc.
    """
    out_pos = pathlib.Path(tempfile.mkdtemp()) / "solution.pos"
    cmd = ["rnx2rtkp"]

    # Minimal usage: rover obs + base obs (or just rover if it already has base)
    cmd += [str(rover_obs_path)]
    if base_obs_path:
        cmd += [str(base_obs_path)]
    if nav_path:
        cmd += [str(nav_path)]

    cmd += ["-o", str(out_pos)]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.stdout, str(out_pos) if out_pos.exists() else None

# =========================
# 4) Parsers
# =========================
def parse_pos(pos_path: str) -> pd.DataFrame:
    """
    Parse RTKLIB .pos into DataFrame with columns: time (UTC-like string in file), lat, lon, h
    RTKLIB default time tag is GPST; we will treat it as naive and compare by string -> datetime.
    """
    rows = []
    with open(pos_path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            # Expect: yyyy/mm/dd hh:mm:ss.sss lat lon height ...
            if len(parts) < 5:
                continue
            ts = parts[0] + " " + parts[1]
            try:
                lat = float(parts[2])
                lon = float(parts[3])
                h = float(parts[4])
            except:
                continue
            rows.append((ts, lat, lon, h))
    df = pd.DataFrame(rows, columns=["ts", "lat", "lon", "h"])
    # parse to datetime (no timezone handling; we just need relative proximity)
    def try_dt(x):
        # RTKLIB format: yyyy/mm/dd HH:MM:SS(.sss)
        fmt = "%Y/%m/%d %H:%M:%S"
        try:
            return datetime.strptime(x[:19], fmt)
        except:
            return None
    df["dt"] = df["ts"].apply(try_dt)
    df = df.dropna(subset=["dt"]).reset_index(drop=True)
    return df

def parse_events_no_header(events_bytes: bytes) -> pd.DataFrame:
    """
    Parse your events .txt with no header.
    Assumptions:
      - Column 0: timestamp string (e.g., 2025-08-09 12:34:56.789 or similar). If not, add your parser.
      - Column 1: image name (optional, keeps whatever string is there)
      - Columns 3,4,5 (0-based) = dN, dE, dU in meters (per your note: columns 4,5,6 1-based).
    You can adjust indexes here if your layout differs.
    """
    txt = events_bytes.decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        parts = ln.replace(",", " ").split()
        if len(parts) < 6:
            # too short for our assumed layout
            continue
        # Take first token as timestamp, second token as image/id
        ts = parts[0]
        name = parts[1]
        try:
            dN = float(parts[3])  # 0-based idx 3 => 4th column
            dE = float(parts[4])  # 5th column
            dU = float(parts[5])  # 6th column
        except:
            continue
        rows.append((ts, name, dN, dE, dU))
    df = pd.DataFrame(rows, columns=["event_ts", "image", "dN", "dE", "dU"])
    # Try parsing several common timestamp formats
    def parse_any(s):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%d %H:%M:%S.%f",
                    "%Y/%m/%d %H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except:
                pass
        # If not parseable, return None
        return None
    df["dt"] = df["event_ts"].apply(parse_any)
    df = df.dropna(subset=["dt"]).reset_index(drop=True)
    return df

# =========================
# 5) Matching + Offset Application
# =========================
def nearest_match(solutions: pd.DataFrame,
                  t: datetime,
                  tolerance_s: float) -> pd.Series | None:
    """Return the closest solution row to t within tolerance_s, else None."""
    if solutions.empty:
        return None
    # Compute absolute |dt - t|
    diffs = (solutions["dt"] - t).abs()
    idx = diffs.idxmin()
    if pd.isna(idx):
        return None
    if diffs.loc[idx] <= timedelta(seconds=tolerance_s):
        return solutions.loc[idx]
    return None

def build_csv(events_df: pd.DataFrame,
              sol_df: pd.DataFrame,
              exposure_delay_s: float,
              tolerance_s: float) -> pd.DataFrame:
    """
    For each event time + delay, find nearest sol sample; apply ENU offsets (N,E,Up) in ECEF;
    return result CSV with image, adjusted lat/lon/alt.
    """
    out_rows = []
    for _, ev in events_df.iterrows():
        # Apply exposure delay: GNSS earlier than shutter → add delay to event time
        t_eff = ev["dt"] + timedelta(seconds=exposure_delay_s)
        nearest = nearest_match(sol_df, t_eff, tolerance_s)
        if nearest is None:
            continue
        lat0, lon0, h0 = float(nearest["lat"]), float(nearest["lon"]), float(nearest["h"])
        dN, dE, dU = float(ev["dN"]), float(ev["dE"]), float(ev["dU"])

        # Apply as vectors in ECEF (correct method)
        lat_adj, lon_adj, h_adj = apply_enu_offset(lat0, lon0, h0, dN, dE, dU)

        out_rows.append({
            "image": ev["image"],
            "event_time": ev["dt"].isoformat(timespec="milliseconds"),
            "matched_time": nearest["dt"].isoformat(timespec="milliseconds"),
            "latitude": lat_adj,
            "longitude": lon_adj,
            "altitude_m": h_adj
        })
    return pd.DataFrame(out_rows)

# =========================
# 6) Streamlit UI
# =========================
st.set_page_config(page_title="PPK Processing with Proper Offsets", layout="wide")
st.title("PPK Processing (RTKLIB + ECEF-vector offsets)")

st.sidebar.header("Inputs")

use_pos = st.sidebar.toggle("I already have a .pos (skip rnx2rtkp)", value=False)

if use_pos:
    pos_file = st.sidebar.file_uploader(".pos from RTKLIB", type=["pos"])
    rnxx_stdout = ""
    pos_path = None
    if pos_file is not None:
        tmp = pathlib.Path(tempfile.mkdtemp())
        pos_path = str(tmp / "solution.pos")
        with open(pos_path, "wb") as f:
            f.write(pos_file.read())
else:
    rover_obs = st.sidebar.file_uploader("Rover RINEX obs", type=["obs", "rnx", "o"])
    base_obs = st.sidebar.file_uploader("Base RINEX obs (optional)", type=["obs", "rnx", "o"])
    nav_rnx  = st.sidebar.file_uploader("Navigation file (optional)", type=["nav", "rnx", "n"])

events_file = st.sidebar.file_uploader("Events file (.txt, no header; cols 4..6 = N,E,Up m)",
                                       type=["txt", "csv"])

exposure_delay_s = st.sidebar.number_input("Exposure delay to add (s)", value=0.062, step=0.001, format="%.3f")
match_tol_s       = st.sidebar.number_input("Match tolerance (s)", value=2.0, step=0.1, format="%.1f")

run_btn = st.sidebar.button("Run Processing")

st.sidebar.caption("Offsets are applied as ENU vectors in ECEF **before** converting back to lat/lon/h.")

if run_btn:
    # 1) Get .pos (either provided or run rnx2rtkp)
    rnxx_stdout = ""
    pos_path = None

    if use_pos:
        if not pos_file:
            st.error("Please upload a .pos file.")
            st.stop()
    else:
        if rover_obs is None:
            st.error("Please upload Rover RINEX.")
            st.stop()

        tmpdir = pathlib.Path(tempfile.mkdtemp())
        r_path = tmpdir / (rover_obs.name or "rover.obs")
        with open(r_path, "wb") as f:
            f.write(rover_obs.read())

        b_path = None
        if base_obs is not None:
            b_path = tmpdir / (base_obs.name or "base.obs")
            with open(b_path, "wb") as f:
                f.write(base_obs.read())

        n_path = None
        if nav_rnx is not None:
            n_path = tmpdir / (nav_rnx.name or "brdc.nav")
            with open(n_path, "wb") as f:
                f.write(nav_rnx.read())

        st.info("Running rnx2rtkp…")
        rnxx_stdout, pos_path = run_rnx2rtkp(str(r_path), str(b_path) if b_path else None, str(n_path) if n_path else None)

    # 2) Show rnx2rtkp output (if any)
    with st.expander("RTKLIB stdout / stderr"):
        st.text(rnxx_stdout if rnxx_stdout else "(no output)")

    if not pos_path or not pathlib.Path(pos_path).exists():
        st.error("PPK failed or .pos not created.")
        st.stop()

    # 3) Parse .pos
    sol_df = parse_pos(pos_path)
    if sol_df.empty:
        st.error("No valid epochs parsed from .pos.")
        st.stop()

    # 4) Parse events
    if events_file is None:
        st.error("Please upload the events .txt.")
        st.stop()
    events_df = parse_events_no_header(events_file.read())
    if events_df.empty:
        st.error("Could not parse events (check columns and timestamp format).")
        st.stop()

    # 5) Build output CSV
    result_df = build_csv(events_df, sol_df, exposure_delay_s, match_tol_s)
    if result_df.empty:
        st.warning("No events matched within tolerance.")
    else:
        st.success(f"Done. {len(result_df)} events matched.")
        st.dataframe(result_df, use_container_width=True)

        # Download
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="ppk_positions.csv", mime="text/csv")

# Helpful notes
with st.expander("Notes"):
    st.markdown("""
- **Offsets** are interpreted as **North, East, Up (meters)** per event row (columns 4–6 in a headerless file).
- We apply them as vectors in the local ENU frame **at the matched epoch**, by converting to ECEF,
  adding the vector, then converting back to geodetic (lat/lon/ellipsoidal height).
- **Exposure delay** (default 0.062 s) is **added** to the event timestamp, because GNSS logging is earlier than the shutter.
- Time matching uses **nearest epoch** within the tolerance (default 2 s).
""")





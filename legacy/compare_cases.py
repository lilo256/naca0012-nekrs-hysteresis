#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_cases.py — Multi-case comparison / overlay post-processor for NekRS
===========================================================================
Overlays and compares an arbitrary set of NekRS RANS cases (here: Series A,
Cases A/B/C — the xfirstu wall-normal resolution sweep for the NACA 0012 at
Re = 1e6, alpha = 10 deg, k-tau, N = 3).

It produces four publication-ready figures:
    1. Cp vs x/c        (upper + lower surface, all cases overlaid)
    2. Cf vs x/c        (with the Cf = 0 separation line, all cases overlaid)
    3. Convergence of Cl and Cd vs convective time (instantaneous + running mean)
    4. y+ vs x/c        (and a distribution/violin summary), all cases overlaid

Design goals
------------
- ONE place to configure everything (the CONFIG block below).
- Schema-tolerant readers: columns are resolved by *name aliases* first, with a
  positional-index fallback, so it works whether your files have headers or not.
- Restart-robust: the forces reader detects NekRS time resets (the internal
  counter restarts on every `startFrom`) and keeps a clean, monotone series
  (last-write-wins per time), so appended `drag_results.dat` files are handled.
- Rotation-robust surface split: upper/lower surfaces are separated via a PCA
  (SVD) chord-axis recovery, so it does not matter that the geometric incidence
  is baked into the mesh — no hard-coded rotation angle is needed.
- Fails soft: if a given data product is missing for a case, that case is
  skipped for that panel with a warning; the rest still plot.

Dependencies: numpy, pandas, matplotlib  (no scipy / sklearn required).

>>> ADAPT HERE <<<  Two things you may need to set once you share real data:
    (a) FORCE_SCHEMA / FORCE_COL_INDEX  — column layout of drag_results.dat
    (b) SURF_SCHEMA                     — column headers of cf_cp_*.csv
The script prints exactly which columns it resolved, so mismatches are obvious.
"""

from __future__ import annotations

import glob
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib

matplotlib.use("Agg")  # headless-safe; remove if you want interactive windows
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG  —  edit this block only
# =============================================================================

# Anchor every relative path below to this file's own folder, not to the
# process's current working directory. Without this, the script only finds
# Baseline/B_xfirst_2e-5/C_xfirst_3e-5 (and only writes comparison_figures/
# where you expect it) when launched from a terminal already cd'd into
# serieA/. An IDE "run" button, a differently-cd'd terminal, etc. use whatever
# cwd they happen to have, silently reading/writing the wrong folder instead
# of failing loudly.
SCRIPT_DIR = Path(__file__).resolve().parent

# --- The cases to compare. Point `path` at each case's output folder. ---------
CASES = [
    dict(label="Case A  (a7_b20)", key="A",
         path=str(SCRIPT_DIR / "a7_b20"), color="#1f4e79", ls="-",  marker=None),
    dict(label="Case B  (a15_b10)", key="B",
         path=str(SCRIPT_DIR / "a15_b10"), color="#c00000", ls="--", marker=None),
    dict(label="Case C  (a20_b30)", key="C",
         path=str(SCRIPT_DIR / "a20_b30"), color="#2e7d32", ls="-.", marker=None),
]

CHORD = 1.0                 # chord length in mesh units (for x/c normalization)
TIME_MIN = 20.0             # convective-time cutoff for STATISTICS (mean over t > TIME_MIN)
N_XC_BINS = 200             # number of x/c bins for surface averaging
OUTPUT_DIR = str(SCRIPT_DIR / "comparison_figures")
DPI = 300
SAVE_PDF = True             # also write vector PDFs alongside PNGs

# --- Impulsive-start transient handling --------------------------------------
# A cold start from a zero field produces an enormous pressure impulse in the
# first few steps: |Cl| can reach O(1e8). Left in, it destroys the autoscale and
# the physical signal collapses onto a flat line at y=0. Samples with
# t < STARTUP_SKIP are discarded outright as non-physical startup transient.
STARTUP_SKIP = 0.05         # convective time; set 0.0 to keep everything
SPIKE_WARN_LEVEL = 50.0     # warn if |Cl| exceeds this anywhere in the record

# Axis limits for the convergence figure. None -> robust percentile autoscale
# (ignores residual outliers). Set explicitly to force a window, e.g. (0.95,1.10).
YLIM_CL = None
YLIM_CD = None
ROBUST_PCTL = (1.0, 99.0)   # percentiles used for the robust autoscale
ROBUST_PAD = 0.25           # fractional padding added around that range

# Reference (Martin Ferrer thesis, Mesh 7, N=2) for annotation on force plots.
REFERENCE = dict(label="Martin Ferrer (ref.)", CL=1.028, CD=0.0205)

# --- File name patterns -------------------------------------------------------
FORCES_FILE = "drag_results.dat"                 # relative to each case path
WALLDIST_GLOB = "wall_distance*"                 # optional, for y+ from real dy1

# --- Column resolution --------------------------------------------------------
# NekRS `drag_results.dat` stores RAW FORCE COMPONENTS, not coefficients:
#   tstep time dragX dragY dragZ dragPX dragPY dragPZ dragVX dragVY dragVZ
#            └── total ──┘ └─ pressure (form) ─┘ └─ viscous (friction) ─┘
# Cl/Cd are derived below. Aliases are case-insensitive; a legacy file that
# already carries `cl`/`cd` columns is still accepted (see load_forces).
FORCE_SCHEMA = dict(
    time=["time", "t", "sim_time"],
    fx=["dragx", "drag_x", "fx"],
    fy=["dragy", "drag_y", "fy"],
    fpx=["dragpx", "drag_px", "fpx"],
    fpy=["dragpy", "drag_py", "fpy"],
    fvx=["dragvx", "drag_vx", "fvx"],
    fvy=["dragvy", "drag_vy", "fvy"],
    # legacy / already-reduced files
    cl=["cl", "c_l", "lift"],
    cd=["cd", "c_d", "drag"],
)
FORCE_COL_INDEX = dict(time=1, fx=2, fy=3, fpx=5, fpy=6, fvx=8, fvy=9)

# Scale applied to the raw force components to obtain coefficients:
#     C = FORCE_NORM * F
# If the UDF already divides by 0.5*rho*U_inf^2*A, leave this at 1.0. If the
# file holds dimensional forces, set FORCE_NORM = 1/(0.5*rho*U^2*c*span).
# The script prints the normalization implied by REFERENCE so a wrong scale is
# immediately visible (see the calibration hint in the summary table).
FORCE_NORM = 1.0

# Incidence used to rotate body-frame forces into the wind frame [degrees].
#
#   *** LEAVE THIS AT 0.0 FOR THIS MESH FAMILY. ***
#
# The geometric AoA is baked into the mesh: the freestream is along +X and the
# airfoil is already tilted, so dragX IS the drag and dragY IS the lift.
# Setting this to 10 rotates a large slice of lift into the drag channel and
# inflates Cd by ~10x (0.0207 -> 0.197) while leaving Cl deceptively near 1.0.
# Only set this non-zero if a FUTURE mesh keeps the chord aligned with X.
# The rotation diagnostic in print_summary() re-checks this at every run.
AOA_ROT_DEG = 0.0

# Sign convention: set to -1 if the UDF reports the force ON the fluid rather
# than on the body (drag would then come out negative).
FORCE_SIGN = +1.0

# Surface CSVs: several naming conventions are tried in order. If none match,
# the loader lists what it actually found so the pattern can be fixed.
SURF_GLOBS = [
    "resultats_csv/*.csv",              # <- actual layout in this project
    "resultats_csv/**/*.csv",
    "cf_cp_time_*_rank_*.csv",
    "cf_cp_*.csv",
    "*cf_cp*.csv",
    "surface_*.csv",
]
# Files matching any of these are NOT surface dumps and are skipped.
SURF_EXCLUDE = ["wall_distance", "final_summary", "summary", "forces"]

SURF_SCHEMA = dict(
    x=["x", "x_coord", "xc", "coord_x"],
    y=["y", "y_coord", "yc", "coord_y"],
    cp=["cp", "c_p", "pressure_coeff", "cp_mean"],
    cf=["cf", "c_f", "skin_friction", "cf_mean"],       # scalar, if present
    cfx=["cf_x", "cfx", "tau_x", "tauw_x"],             # vector components
    cfy=["cf_y", "cfy", "tau_y", "tauw_y"],
    yplus=["yplus", "y_plus", "yp", "y+"],
)

# How to reduce a skin-friction VECTOR (cf_x, cf_y) to the scalar Cf plotted
# against x/c:
#   "auto"      - use cf_x directly if cf_y is identically zero (the UDF already
#                 wrote a scalar there); otherwise project onto the chord axis
#   "project"   - always project (cf . chord_unit_vector), signed
#   "magnitude" - always |cf|, unsigned (loses the separation sign!)
CF_VECTOR_MODE = "auto"

# --- y+ reconstruction from wall_distance (optional) --------------------------
# The surface dumps carry no y+ column here, so it is rebuilt from the wall
# distance and the local skin friction:
#     u_tau = U_inf * sqrt(|Cf| / 2)   ;   y+ = dy1 * u_tau / nu = dy1*sqrt(|Cf|/2)*Re
# Requires REYNOLDS and a wall_distance file with x, y and a distance column.
REYNOLDS = 1.0e6
WALLDIST_SCHEMA = dict(
    x=["x", "x_coord", "coord_x"],
    y=["y", "y_coord", "coord_y"],
    d=["wall_distance", "dist", "distance", "dy1", "d", "wd", "y1"],
)
# wall_distance dumps contain every volume node, not just near-wall ones (a
# brute-force nearest-neighbour match against all of them blows up memory:
# N_surface x N_volume doubles). Restrict candidates to points actually near
# the wall before the KD-tree query, same thresholds as check_yplus.py.
WALLDIST_D_MIN = 1e-12
WALLDIST_D_MAX = 1e-2
WALLDIST_N_NEIGH = 12

# =============================================================================
# Plot style (publication)
# =============================================================================
def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "CMU Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "axes.linewidth": 1.0,
        "legend.fontsize": 10.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "lines.linewidth": 1.8,
        "grid.color": "0.85",
        "grid.linewidth": 0.7,
        "figure.constrained_layout.use": True,
    })


# =============================================================================
# Small helpers
# =============================================================================
def _resolve_columns(df: pd.DataFrame, schema: dict) -> dict:
    """Map logical names -> actual df column, using case-insensitive aliases."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    out = {}
    for logical, aliases in schema.items():
        hit = next((lower[a] for a in aliases if a in lower), None)
        out[logical] = hit
    return out


def _looks_headerless(path: Path) -> bool:
    """Heuristic: first non-comment line is all-numeric -> no header."""
    with open(path, "r", errors="ignore") as fh:
        for raw in fh:
            s = raw.strip()
            if not s or s[0] in "#%":
                continue
            toks = s.replace(",", " ").split()
            try:
                [float(t) for t in toks]
                return True          # parsed as numbers => header-less
            except ValueError:
                return False          # has text => header row
    return True


_TIME_RE = re.compile(r"time_([0-9]+(?:[_.][0-9]+)?)", re.IGNORECASE)


def _time_from_name(name: str) -> float | None:
    """Extract real sim time from names like cf_cp_time_0_873324_rank_1.csv."""
    m = _TIME_RE.search(name)
    if not m:
        return None
    return float(m.group(1).replace("_", "."))


def _robust_ylim(series, pad=ROBUST_PAD, pctl=ROBUST_PCTL):
    """
    Percentile-based axis limits, immune to residual startup spikes.
    `series` is a list of 1-D arrays (may contain None / empties).
    """
    vals = [np.asarray(s, float) for s in series if s is not None and len(s)]
    if not vals:
        return None
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = np.percentile(v, pctl)
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        span = max(abs(hi), 1e-9)
    return lo - pad * span, hi + pad * span


def _pca_chord_frame(x: np.ndarray, y: np.ndarray, return_axis: bool = False):
    """
    Rotation-robust chord frame via SVD.
    Returns (s, n) where s is the chordwise coordinate (along max-variance axis)
    and n is the surface-normal-ish coordinate (perpendicular). Sign of s is
    fixed so it increases from LE(min) to TE(max); n>0 is taken as 'upper'.
    If return_axis, also returns the unit chord vector.
    """
    pts = np.column_stack([x, y]).astype(float)
    c = pts.mean(axis=0)
    q = pts - c
    # principal axes
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    e_chord = vt[0] / np.linalg.norm(vt[0])
    e_norm = np.array([-e_chord[1], e_chord[0]])
    s = q @ e_chord
    n = q @ e_norm
    # orient s so it grows LE->TE (assume LE is the point of min x in raw frame)
    if x[np.argmin(s)] > x[np.argmax(s)]:
        s, e_chord = -s, -e_chord
    if return_axis:
        return s, n, e_chord
    return s, n


def _bin_mean(xc: np.ndarray, val: np.ndarray, edges: np.ndarray):
    """Bin `val` by `xc` into `edges`; return (centers, mean, rms) ignoring empties."""
    idx = np.digitize(xc, edges) - 1
    nb = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(nb, np.nan)
    rms = np.full(nb, np.nan)
    for b in range(nb):
        sel = idx == b
        if np.any(sel):
            v = val[sel]
            mean[b] = np.nanmean(v)
            rms[b] = np.nanstd(v)
    return centers, mean, rms


# =============================================================================
# Case container
# =============================================================================
@dataclass
class Case:
    label: str
    key: str
    path: str
    color: str
    ls: str = "-"
    marker: str | None = None

    # loaded products
    t: np.ndarray | None = None
    cl: np.ndarray | None = None
    cd: np.ndarray | None = None
    cdp: np.ndarray | None = None         # form (pressure) drag coefficient
    cdv: np.ndarray | None = None         # friction (viscous) drag coefficient
    surf: pd.DataFrame | None = None      # columns: xc, n, cp, cf, yplus, upper
    cl_conv: float = np.nan
    cd_conv: float = np.nan
    cdp_conv: float = np.nan
    cdv_conv: float = np.nan

    # -- forces -----------------------------------------------------------------
    def load_forces(self):
        f = Path(self.path) / FORCES_FILE
        if not f.exists():
            warnings.warn(f"[{self.key}] no {FORCES_FILE} in {self.path}")
            return
        headerless = _looks_headerless(f)
        try:
            if headerless:
                df = pd.read_csv(f, sep=r"\s+", comment="#", header=None,
                                 engine="python")
                df.columns = list(range(df.shape[1]))
                get = lambda k: (df.iloc[:, FORCE_COL_INDEX[k]].to_numpy(float)
                                 if FORCE_COL_INDEX.get(k) is not None
                                 and FORCE_COL_INDEX[k] < df.shape[1] else None)
                t = get("time")
                fx, fy = get("fx"), get("fy")
                fpx, fpy = get("fpx"), get("fpy")
                fvx, fvy = get("fvx"), get("fvy")
                cl_raw = cd_raw = None
                print(f"[{self.key}] forces: header-less, using positional "
                      f"indices {FORCE_COL_INDEX}")
            else:
                df = pd.read_csv(f, sep=r"\s+", comment="#", engine="python")
                m = _resolve_columns(df, FORCE_SCHEMA)
                col = lambda k: (df[m[k]].to_numpy(float)
                                 if m.get(k) is not None else None)
                t = col("time")
                fx, fy = col("fx"), col("fy")
                fpx, fpy = col("fpx"), col("fpy")
                fvx, fvy = col("fvx"), col("fvy")
                cl_raw, cd_raw = col("cl"), col("cd")
                if t is None:
                    raise KeyError(f"no time column; got {list(df.columns)}")
                if fx is None and cd_raw is None:
                    raise KeyError(
                        "found neither force components (dragX/dragY) nor "
                        f"coefficients (cl/cd); got {list(df.columns)}")
                found = {k: v for k, v in m.items() if v is not None}
                print(f"[{self.key}] forces: resolved {found}")

            # --- derive coefficients ------------------------------------------
            if cd_raw is not None and cl_raw is not None:
                cl, cd = cl_raw, cd_raw
                cdp = cdv = None
                print(f"[{self.key}] using pre-computed Cl/Cd columns")
            else:
                a = np.deg2rad(AOA_ROT_DEG)
                ca, sa = np.cos(a), np.sin(a)
                k = FORCE_SIGN * FORCE_NORM

                def wind(fxc, fyc):
                    """Body -> wind frame. Returns (drag, lift) coefficients."""
                    if fxc is None or fyc is None:
                        return None, None
                    return (k * (fxc * ca + fyc * sa),
                            k * (-fxc * sa + fyc * ca))

                cd, cl = wind(fx, fy)
                cdp, _ = wind(fpx, fpy)
                cdv, _ = wind(fvx, fvy)
                print(f"[{self.key}] derived Cd/Cl from force components "
                      f"(AoA_rot={AOA_ROT_DEG}deg, norm={FORCE_NORM:g}, "
                      f"sign={FORCE_SIGN:+g})")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[{self.key}] failed to read forces: {exc}")
            return

        t, cl, cd, cdp, cdv = self._clean_restarts(t, cl, cd, cdp, cdv)

        # --- discard the impulsive-start transient -----------------------------
        peak = np.nanmax(np.abs(cl)) if cl.size else 0.0
        if peak > SPIKE_WARN_LEVEL:
            print(f"[{self.key}] impulsive-start spike detected: max|Cl| = "
                  f"{peak:.3g} -> discarding t < {STARTUP_SKIP}")
        if STARTUP_SKIP > 0:
            keep = t >= STARTUP_SKIP
            ndrop = int((~keep).sum())
            if ndrop:
                print(f"[{self.key}] dropped {ndrop} startup sample(s) "
                      f"(t < {STARTUP_SKIP})")
            t, cl, cd = t[keep], cl[keep], cd[keep]
            cdp = cdp[keep] if cdp is not None else None
            cdv = cdv[keep] if cdv is not None else None
        if t.size == 0:
            warnings.warn(f"[{self.key}] no samples left after startup skip")
            return

        self.t, self.cl, self.cd, self.cdp, self.cdv = t, cl, cd, cdp, cdv

        # Decomposition identity, checked on the PHYSICAL data only. Run on the
        # raw record it is dominated by print-precision noise in the 1e8 startup
        # spike (a file written to 9 significant digits carries ~0.1 absolute
        # error at 1e8), which looks alarming but is meaningless.
        if cdp is not None and cdv is not None and t.size:
            resid = np.nanmax(np.abs((cdp + cdv) - cd))
            scale = float(np.nanmean(np.abs(cd))) or 1.0
            rel = resid / scale
            flag = "OK" if rel < 1e-6 else "MISMATCH"
            print(f"[{self.key}] decomposition Cd_p + Cd_v == Cd : {flag} "
                  f"(relative {rel:.2e})")

        mask = t >= TIME_MIN
        if not np.any(mask):
            warnings.warn(f"[{self.key}] no samples past TIME_MIN={TIME_MIN}; "
                          f"max t = {t.max():.2f}. Reporting full-series mean.")
            mask = np.ones_like(t, dtype=bool)
        self.cl_conv = float(np.mean(cl[mask]))
        self.cd_conv = float(np.mean(cd[mask]))
        if cdp is not None:
            self.cdp_conv = float(np.mean(cdp[mask]))
        if cdv is not None:
            self.cdv_conv = float(np.mean(cdv[mask]))

    @staticmethod
    def _clean_restarts(t, cl, cd, cdp=None, cdv=None):
        """
        NekRS resets its counter on every startFrom, so an appended
        drag_results.dat can contain non-monotone time. Keep a clean monotone
        series with last-write-wins per (rounded) time.
        """
        order = np.argsort(t, kind="stable")
        # de-duplicate exact/near-equal times, keeping the last occurrence
        key = np.round(t[order], 9)
        _, last_idx = np.unique(key[::-1], return_index=True)
        keep = order[np.sort(len(t) - 1 - last_idx)]
        sub = lambda a: a[keep] if a is not None else None
        return t[keep], cl[keep], cd[keep], sub(cdp), sub(cdv)

    # -- surface (cp / cf / y+) -------------------------------------------------
    def load_surface(self):
        files, used = [], None
        for pat in SURF_GLOBS:
            hits = sorted(glob.glob(str(Path(self.path) / pat)))
            if hits:
                files, used = hits, pat
                break
        if not files:
            # Self-diagnose: show what CSV-like files actually exist, so the
            # pattern can be corrected instead of failing silently.
            here = Path(self.path)
            found = sorted(
                [p.name for p in here.glob("*.csv")][:8]
                + [f"{p.name}/" for p in here.iterdir() if p.is_dir()][:8]
            ) if here.exists() else []
            warnings.warn(
                f"[{self.key}] no surface files matched any of {SURF_GLOBS}.\n"
                f"    Directory contains: {found or '(nothing csv-like)'}\n"
                f"    -> add the correct pattern to SURF_GLOBS.")
            return
        files = [f for f in files
                 if not any(bad in os.path.basename(f).lower()
                            for bad in SURF_EXCLUDE)]
        if not files:
            warnings.warn(f"[{self.key}] pattern '{used}' matched only excluded "
                          f"files ({SURF_EXCLUDE})")
            return
        print(f"[{self.key}] surface: pattern '{used}' matched {len(files)} file(s)")
        frames = []
        for fp in files:
            tt = _time_from_name(os.path.basename(fp))
            if tt is not None and tt < TIME_MIN:
                continue
            try:
                d = pd.read_csv(fp)
            except Exception:  # noqa: BLE001
                continue
            if d.empty:
                continue
            frames.append(d)
        if not frames:
            warnings.warn(f"[{self.key}] all surface files empty or pre-transient")
            return
        raw = pd.concat(frames, ignore_index=True)
        # Restart-appended dumps can carry an embedded duplicate header row
        # mid-file (e.g. a literal "x,y,z,cf_x,..." line where a run resumed
        # and rewrote the header). Coerce to numeric and drop any such stray
        # rows instead of crashing downstream on a string value.
        n0 = len(raw)
        raw = raw.apply(pd.to_numeric, errors="coerce").dropna(how="any")
        if len(raw) < n0:
            print(f"[{self.key}] surface: dropped {n0 - len(raw)} malformed "
                  f"row(s) (likely embedded restart header)")
        m = _resolve_columns(raw, SURF_SCHEMA)
        need = [k for k in ("x", "cp") if m[k] is None]
        if need:
            warnings.warn(f"[{self.key}] surface missing {need}; cols={list(raw.columns)}")
            return
        found = {k: v for k, v in m.items() if v is not None}
        print(f"[{self.key}] surface: resolved {found} "
              f"({len(files)} file(s), {len(raw)} points)")

        x = raw[m["x"]].to_numpy(float)
        y = (raw[m["y"]].to_numpy(float) if m["y"] is not None
             else np.zeros_like(x))
        s, n, e_chord = _pca_chord_frame(x, y, return_axis=True)
        xc = (s - s.min()) / CHORD
        surf = pd.DataFrame(dict(x=x, y=y, xc=xc, n=n, upper=(n >= 0)))
        surf["cp"] = raw[m["cp"]].to_numpy(float)

        # --- skin friction: scalar column, or reduce the (cf_x, cf_y) vector ---
        if m["cf"] is not None:
            cf = raw[m["cf"]].to_numpy(float)
            how = "scalar column"
        elif m["cfx"] is not None:
            cfx = raw[m["cfx"]].to_numpy(float)
            cfy = (raw[m["cfy"]].to_numpy(float) if m["cfy"] is not None
                   else np.zeros_like(cfx))
            y_is_zero = np.allclose(cfy, 0.0, atol=1e-14)
            mode = CF_VECTOR_MODE
            if mode == "auto":
                mode = "scalar_x" if y_is_zero else "project"
            if mode == "scalar_x":
                cf = cfx
                how = "cf_x taken as scalar (cf_y identically zero)"
            elif mode == "magnitude":
                cf = np.hypot(cfx, cfy)
                how = "|cf| magnitude (UNSIGNED - separation sign lost)"
            else:
                cf = cfx * e_chord[0] + cfy * e_chord[1]
                how = "signed projection onto chord axis"
            print(f"[{self.key}] Cf from vector: {how}")
        else:
            cf = np.full_like(x, np.nan)
            how = "not found"
        surf["cf"] = cf

        # --- y+ : direct column if present, else rebuild from wall_distance ----
        if m["yplus"] is not None:
            surf["yplus"] = raw[m["yplus"]].to_numpy(float)
        else:
            surf["yplus"] = self._yplus_from_walldist(x, y, cf)
        self.surf = surf

    def _yplus_from_walldist(self, x, y, cf):
        """
        y+ = dy1 * u_tau / nu  with  u_tau = U_inf*sqrt(|Cf|/2), nu = U_inf*c/Re
             => y+ = dy1 * sqrt(|Cf|/2) * Re      (U_inf = c = 1)
        The wall distance is matched to each surface point by nearest neighbour.
        """
        hits = sorted(glob.glob(str(Path(self.path) / WALLDIST_GLOB)))
        hits = [h for h in hits if h.lower().endswith(".csv")]
        if not hits:
            return np.full_like(x, np.nan)
        try:
            wd = pd.concat([pd.read_csv(h) for h in hits], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[{self.key}] wall_distance unreadable: {exc}")
            return np.full_like(x, np.nan)
        mw = _resolve_columns(wd, WALLDIST_SCHEMA)
        if mw["d"] is None or mw["x"] is None:
            warnings.warn(f"[{self.key}] wall_distance: no distance/x column; "
                          f"cols={list(wd.columns)} -> y+ unavailable")
            return np.full_like(x, np.nan)
        wx = wd[mw["x"]].to_numpy(float)
        wy = (wd[mw["y"]].to_numpy(float) if mw["y"] is not None
              else np.zeros_like(wx))
        wdist = wd[mw["d"]].to_numpy(float)

        # wall_distance dumps carry every volume node (hundreds of thousands),
        # most of them far from the wall. Restrict to near-wall candidates
        # first, then match via a KD-tree — a brute-force N_surface x N_volume
        # distance matrix here is what previously blew up memory (~100+ GiB).
        near = (wdist > WALLDIST_D_MIN) & (wdist < WALLDIST_D_MAX)
        if near.sum() == 0:
            warnings.warn(f"[{self.key}] wall_distance: no point within "
                          f"({WALLDIST_D_MIN:g}, {WALLDIST_D_MAX:g}); "
                          f"y+ unavailable")
            return np.full_like(x, np.nan)
        wx, wy, wdist = wx[near], wy[near], wdist[near]

        tree = cKDTree(np.column_stack([wx, wy]))
        k = min(WALLDIST_N_NEIGH, len(wdist))
        _, idx = tree.query(np.column_stack([x, y]), k=k)
        dy1 = wdist[idx].min(axis=1) if k > 1 else wdist[idx]
        yp = dy1 * np.sqrt(np.abs(cf) / 2.0) * REYNOLDS
        print(f"[{self.key}] y+ reconstructed from wall_distance "
              f"(Re={REYNOLDS:g}): mean={np.nanmean(yp):.3f}, "
              f"max={np.nanmax(yp):.3f}")
        return yp

    # -- binned mean of a surface quantity, split by surface --------------------
    def surface_profile(self, quantity: str, upper: bool):
        if self.surf is None or self.surf[quantity].isna().all():
            return None
        sel = self.surf[self.surf["upper"] == upper]
        if sel.empty:
            return None
        edges = np.linspace(0.0, np.nanmax(self.surf["xc"]), N_XC_BINS + 1)
        return _bin_mean(sel["xc"].to_numpy(), sel[quantity].to_numpy(), edges)


# =============================================================================
# Plotters
# =============================================================================
def plot_cp(cases, outdir):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plotted = False
    for c in cases:
        for upper, alpha, lab in ((True, 1.0, "upper"), (False, 0.55, "lower")):
            prof = c.surface_profile("cp", upper)
            if prof is None:
                continue
            xc, mean, _ = prof
            ax.plot(xc, mean, color=c.color, ls=c.ls, alpha=alpha,
                    label=f"{c.label} — {lab}")
            plotted = True
    if not plotted:
        plt.close(fig)
        warnings.warn("Cp: nothing to plot"); return
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$C_p$")
    ax.set_title(r"Surface pressure coefficient — case comparison")
    ax.invert_yaxis()                     # aerodynamic convention
    ax.grid(True, alpha=0.5)
    ax.legend(loc="best", ncol=1)
    _save(fig, outdir, "cp_comparison")


def plot_cf(cases, outdir):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plotted = False
    for c in cases:
        for upper, alpha, lab in ((True, 1.0, "upper"), (False, 0.55, "lower")):
            prof = c.surface_profile("cf", upper)
            if prof is None:
                continue
            xc, mean, _ = prof
            ax.plot(xc, mean, color=c.color, ls=c.ls, alpha=alpha,
                    label=f"{c.label} — {lab}")
            plotted = True
    if not plotted:
        plt.close(fig)
        warnings.warn("Cf: nothing to plot"); return
    ax.axhline(0, color="k", lw=1.0, ls=":", zorder=1,
               label=r"$C_f=0$ (separation)")
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$C_f$")
    ax.set_title(r"Skin-friction coefficient — case comparison")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="best", ncol=1)
    _save(fig, outdir, "cf_comparison")


def plot_convergence(cases, outdir):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True)
    any_data = False
    for c in cases:
        if c.t is None:
            continue
        any_data = True
        run_cl = np.cumsum(c.cl) / np.arange(1, len(c.cl) + 1)
        run_cd = np.cumsum(c.cd) / np.arange(1, len(c.cd) + 1)
        a1.plot(c.t, c.cl, color=c.color, ls=c.ls, alpha=0.30, lw=1.0)
        a1.plot(c.t, run_cl, color=c.color, ls=c.ls, lw=2.0,
                label=f"{c.label}  (running mean $C_L$={c.cl_conv:.4f})")
        a2.plot(c.t, c.cd, color=c.color, ls=c.ls, alpha=0.30, lw=1.0)
        a2.plot(c.t, run_cd, color=c.color, ls=c.ls, lw=2.0,
                label=f"{c.label}  (running mean $C_D$={c.cd_conv:.5f})")
    if not any_data:
        plt.close(fig)
        warnings.warn("Convergence: no force data"); return

    if REFERENCE.get("CL"):
        a1.axhline(REFERENCE["CL"], color="0.35", ls=(0, (4, 3)), lw=1.2,
                   label=f"{REFERENCE['label']} $C_L$={REFERENCE['CL']}")
    if REFERENCE.get("CD"):
        a2.axhline(REFERENCE["CD"], color="0.35", ls=(0, (4, 3)), lw=1.2,
                   label=f"{REFERENCE['label']} $C_D$={REFERENCE['CD']}")
    # --- axis limits: manual override, else robust percentile autoscale -------
    cl_lim = YLIM_CL or _robust_ylim([c.cl for c in cases])
    cd_lim = YLIM_CD or _robust_ylim([c.cd for c in cases])
    if cl_lim:
        # keep the reference line inside the frame
        if REFERENCE.get("CL"):
            cl_lim = (min(cl_lim[0], REFERENCE["CL"]),
                      max(cl_lim[1], REFERENCE["CL"]))
        a1.set_ylim(*cl_lim)
    if cd_lim:
        if REFERENCE.get("CD"):
            cd_lim = (min(cd_lim[0], REFERENCE["CD"]),
                      max(cd_lim[1], REFERENCE["CD"]))
        a2.set_ylim(*cd_lim)

    for ax in (a1, a2):
        ax.axvline(TIME_MIN, color="0.6", ls=":", lw=1.0)
        ax.grid(True, alpha=0.5)
    a1.set_ylabel(r"$C_L$")
    a2.set_ylabel(r"$C_D$")
    a2.set_xlabel(r"Convective time  $t\,U_\infty/c$")
    a1.set_title(r"Integral-force convergence — case comparison "
                 rf"(transient cutoff $t={TIME_MIN:g}$)")
    a1.legend(loc="best", fontsize=9)
    a2.legend(loc="best", fontsize=9)
    _save(fig, outdir, "convergence_cl_cd")


def plot_yplus(cases, outdir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                                 gridspec_kw=dict(width_ratios=[2.4, 1]))
    dist, labels, colors = [], [], []
    any_profile = False
    for c in cases:
        prof_u = c.surface_profile("yplus", True)
        prof_l = c.surface_profile("yplus", False)
        for prof in (prof_u, prof_l):
            if prof is None:
                continue
            xc, mean, _ = prof
            a1.plot(xc, mean, color=c.color, ls=c.ls, lw=1.8)
            any_profile = True
        if c.surf is not None and not c.surf["yplus"].isna().all():
            v = c.surf["yplus"].to_numpy()
            dist.append(v[np.isfinite(v)])
            labels.append(c.key)
            colors.append(c.color)
    if any_profile:
        # one legend entry per case (not per surface)
        for c in cases:
            a1.plot([], [], color=c.color, ls=c.ls, lw=1.8, label=c.label)
    a1.axhline(1.0, color="k", ls="--", lw=1.2, label=r"$y^+=1$ ceiling")
    a1.set_xlabel(r"$x/c$")
    a1.set_ylabel(r"$y^+$")
    a1.set_title(r"$y^+$ distribution along chord")
    a1.grid(True, alpha=0.5)
    a1.legend(loc="best", fontsize=9)

    if dist:
        parts = a2.violinplot(dist, showmeans=True, showextrema=True)
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.45)
        a2.set_xticks(range(1, len(labels) + 1))
        a2.set_xticklabels(labels)
        a2.axhline(1.0, color="k", ls="--", lw=1.2)
        a2.set_ylabel(r"$y^+$")
        a2.set_title(r"$y^+$ spread")
        a2.grid(True, axis="y", alpha=0.5)
    else:
        a2.axis("off")

    if not any_profile and not dist:
        plt.close(fig)
        warnings.warn("y+: nothing to plot"); return
    _save(fig, outdir, "yplus_comparison")


def plot_drag_decomposition(cases, outdir):
    """Form vs friction drag histories — only if the raw components were found."""
    have = [c for c in cases if c.cdp is not None and c.cdv is not None]
    if not have:
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True)
    for c in have:
        run_p = np.cumsum(c.cdp) / np.arange(1, len(c.cdp) + 1)
        run_v = np.cumsum(c.cdv) / np.arange(1, len(c.cdv) + 1)
        a1.plot(c.t, c.cdp, color=c.color, ls=c.ls, alpha=0.30, lw=1.0)
        a1.plot(c.t, run_p, color=c.color, ls=c.ls, lw=2.0,
                label=f"{c.label}  ($C_{{D,p}}$={c.cdp_conv:.5f})")
        a2.plot(c.t, c.cdv, color=c.color, ls=c.ls, alpha=0.30, lw=1.0)
        a2.plot(c.t, run_v, color=c.color, ls=c.ls, lw=2.0,
                label=f"{c.label}  ($C_{{D,v}}$={c.cdv_conv:.5f})")
    p_lim = _robust_ylim([c.cdp for c in have])
    v_lim = _robust_ylim([c.cdv for c in have])
    if p_lim:
        a1.set_ylim(*p_lim)
    if v_lim:
        a2.set_ylim(*v_lim)
    for ax in (a1, a2):
        ax.axvline(TIME_MIN, color="0.6", ls=":", lw=1.0)
        ax.grid(True, alpha=0.5)
        ax.legend(loc="best", fontsize=9)
    a1.set_ylabel(r"$C_{D,p}$  (form)")
    a2.set_ylabel(r"$C_{D,v}$  (friction)")
    a2.set_xlabel(r"Convective time  $t\,U_\infty/c$")
    a1.set_title("Drag decomposition — form vs friction")
    _save(fig, outdir, "drag_decomposition")


def _save(fig, outdir, stem):
    png = Path(outdir) / f"{stem}.png"
    fig.savefig(png, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(Path(outdir) / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png}")


# =============================================================================
# Summary table
# =============================================================================
def print_summary(cases):
    print("\n" + "=" * 84)
    print(f"{'Case':<26}{'CL':>10}{'CD':>12}{'CD_p':>12}{'CD_v':>12}{'y+max':>10}")
    print("-" * 84)
    for c in cases:
        ypmax = (np.nanmax(c.surf['yplus']) if (c.surf is not None
                 and not c.surf['yplus'].isna().all()) else np.nan)
        fmt = lambda v, n: f"{v:.{n}f}" if np.isfinite(v) else "n/a"
        print(f"{c.key + '  ' + c.label:<26}"
              f"{fmt(c.cl_conv,4):>10}{fmt(c.cd_conv,5):>12}"
              f"{fmt(c.cdp_conv,5):>12}{fmt(c.cdv_conv,5):>12}"
              f"{fmt(ypmax,2):>10}")
    if REFERENCE.get("CL"):
        print(f"{'REF ' + REFERENCE['label']:<26}"
              f"{REFERENCE['CL']:>10}{REFERENCE['CD']:>12}{'-':>12}"
              f"{'-':>12}{'-':>10}")
    print("=" * 84)

    # --- calibration diagnostic ----------------------------------------------
    # A wrong FORCE_NORM scales Cd and Cl by the SAME factor.
    # A wrong AOA_ROT_DEG mixes lift into drag: Cd is badly wrong while Cl stays
    # deceptively close to target. Distinguish the two before suggesting a fix.
    if REFERENCE.get("CD") and REFERENCE.get("CL"):
        ok = [c for c in cases
              if np.isfinite(c.cd_conv) and np.isfinite(c.cl_conv) and c.cd_conv]
        if ok:
            rd = float(np.mean([REFERENCE["CD"] / c.cd_conv for c in ok]))
            rl = float(np.mean([REFERENCE["CL"] / c.cl_conv for c in ok]))
            same = abs(rd - rl) / max(abs(rd), 1e-12) < 0.10

            if any(c.cd_conv < 0 for c in ok):
                print("\n  ** Negative drag -> set FORCE_SIGN = -1.")
            elif same and not (0.9 < rd < 1.1):
                print(f"\n  ** Cd and Cl are BOTH off by the same factor "
                      f"~{rd:.4g}x.\n     That is a normalization problem: try "
                      f"FORCE_NORM = {FORCE_NORM * rd:.6g} (currently "
                      f"{FORCE_NORM:g}).")
            elif not same:
                # Recover the rotation angle implied by the data vs. reference.
                # The transform Cd = fx.cos a + fy.sin a, Cl = -fx.sin a + fy.cos a
                # is a rotation of the raw vector by -a, so the observed angle is
                #     phi_obs = phi_raw - a_current
                # and the angle that would land on the reference is
                #     a_correct = phi_raw - phi_ref = a_current + (phi_obs - phi_ref)
                th_ref = np.arctan2(REFERENCE["CL"], REFERENCE["CD"])
                dtheta = float(np.mean(
                    [np.rad2deg(np.arctan2(c.cl_conv, c.cd_conv) - th_ref)
                     for c in ok]))
                suggested = AOA_ROT_DEG + dtheta
                print(f"\n  ** Cd is off by {rd:.4g}x but Cl only by {rl:.4g}x.")
                print(f"     Unequal factors = lift leaking into the drag "
                      f"channel, i.e. a ROTATION error, not a scale error.")
                print(f"     Do NOT change FORCE_NORM. Set AOA_ROT_DEG = "
                      f"{suggested:.2f} (currently {AOA_ROT_DEG:g}).")
            else:
                print(f"\n  Calibration OK: Cd within {abs(1-1/rd)*100:.1f}% and "
                      f"Cl within {abs(1-1/rl)*100:.1f}% of the reference.")
    print()


# =============================================================================
# Main
# =============================================================================
def main(argv=None):
    set_pub_style()
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    cases = [Case(**{k: v for k, v in d.items()}) for d in CASES]
    for c in cases:
        print(f"\n--- loading {c.key}: {c.path} ---")
        c.load_forces()
        c.load_surface()

    print("\n--- plotting ---")
    plot_cp(cases, outdir)
    plot_cf(cases, outdir)
    plot_convergence(cases, outdir)
    plot_drag_decomposition(cases, outdir)
    plot_yplus(cases, outdir)

    print_summary(cases)
    print(f"All figures in: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

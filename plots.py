"""Figures generated from real data: ``sms figures`` and the summary figures.

Brightness-temperature images use a grey ramp with cold = white (the infrared convention); signed
differences use a blue-red map with a grey midpoint; line charts use at most three colours.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .acquire import parse_abi_name, read_csv, read_manifest  # noqa: E402
from .config import from_portable, manifest_dir, reports_dir  # noqa: E402
from .preprocess import brightness_temperature, read_cmip, read_scan  # noqa: E402

TEXT = "#191919"
TEXT_2 = "#666666"
GRID = "#dddddd"
SERIES = ("#257de0", "#f47230", "#00b894")  # categorical slots 1-3
DIVERGING = LinearSegmentedColormap.from_list(
    "sms_diverging", ["#0e3d7e", "#257de0", "#9dc8f7", "#efefef", "#f5a3a3", "#ec4647", "#8e1d1d"])
BT_CMAP = "Greys"  # low values (cold) white, high values (warm) black
BT_LIMITS_K = (190.0, 310.0)


def _style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "text.color": TEXT, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
        "legend.frameon": False, "lines.linewidth": 1.5,
    })


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_window_strip(core_bt_K: np.ndarray, t_fractions: Sequence[float], path: Path,
                      ctx_before: np.ndarray | None = None, ctx_after: np.ndarray | None = None,
                      ctx_t: tuple[float | None, float | None] = (None, None), title: str = "") -> Path:
    """All frames of one window patch in time order: I_-1, I_0, nine hidden frames, I_1, I_2."""
    _style()
    frames: list[tuple[str, np.ndarray | None]] = [(f"$I_{{-1}}$ t={ctx_t[0]:.3f}" if ctx_before is not None
                                                    else "$I_{-1}$ absent", ctx_before)]
    n = len(t_fractions)
    for k, (img, t) in enumerate(zip(core_bt_K, t_fractions)):
        label = "$I_0$" if k == 0 else ("$I_1$" if k == n - 1 else f"hidden {k}")
        frames.append((f"{label} t={t:.3f}", img))
    frames.append((f"$I_2$ t={ctx_t[1]:.3f}" if ctx_after is not None else "$I_2$ absent", ctx_after))
    cols = 7
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.45 * cols, 1.6 * rows + 0.4))
    im = None
    for ax, (label, img) in zip(axes.ravel(), frames):
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(label, fontsize=7)
        if img is None:
            ax.text(0.5, 0.5, "no real frame", ha="center", va="center", fontsize=7, color=TEXT_2,
                    transform=ax.transAxes)
            continue
        im = ax.imshow(img, cmap=BT_CMAP, vmin=BT_LIMITS_K[0], vmax=BT_LIMITS_K[1], interpolation="nearest")
    for ax in axes.ravel()[len(frames):]:
        ax.axis("off")
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
        cb.set_label("K")
    if title:
        fig.suptitle(title, fontsize=9)
    return _save(fig, path)


def plot_calibration_difference(diff_K: np.ndarray, valid: np.ndarray, step_K: float, path: Path, title: str = "") -> Path:
    """Histogram and map of (our brightness temperature - NOAA CMIP)."""
    _style()
    d = diff_K[valid]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.2, 1]})
    lim = max(step_K * 0.75, float(np.abs(d).max()) * 1.05)
    ax1.hist(d, bins=np.linspace(-lim, lim, 81), color=SERIES[0], edgecolor="white", linewidth=0.3)
    for s in (-step_K / 2, step_K / 2):
        ax1.axvline(s, color=TEXT_2, lw=1, ls="--")
    ax1.text(step_K / 2, ax1.get_ylim()[1] * 0.92, r" $\pm\Delta/2$", color=TEXT_2, fontsize=8)
    ax1.set_xlabel(r"$T_B^{\mathrm{ours}} - T_B^{\mathrm{CMIP}}$ (K)")
    ax1.set_ylabel("pixels")
    ax1.grid(axis="y")
    show = np.where(valid, diff_K, np.nan)
    im = ax2.imshow(show, cmap=DIVERGING, vmin=-lim, vmax=lim, interpolation="nearest")
    ax2.set_xticks([])
    ax2.set_yticks([])
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    cb.set_label("K")
    if title:
        fig.suptitle(title, fontsize=9)
    return _save(fig, path)


def plot_coverage_check(bt_K: np.ndarray, storm_rc: tuple[float, float], margin_px: int, path: Path, title: str) -> Path:
    """A coverage sample with the storm position and the edge margin of the verification rule."""
    _style()
    H, W = bt_K.shape
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(bt_K, cmap=BT_CMAP, vmin=BT_LIMITS_K[0], vmax=BT_LIMITS_K[1], interpolation="nearest")
    ax.add_patch(plt.Rectangle((margin_px - 0.5, margin_px - 0.5), W - 2 * margin_px, H - 2 * margin_px,
                               fill=False, ec=SERIES[0], lw=1.2, ls="--", label=f"{margin_px}-pixel margin"))
    r, c = storm_rc
    ax.plot(c, r, marker="o", ms=8, mfc="none", mec=SERIES[1], mew=2, ls="none", label="storm position")
    ax.set_xlim(min(-0.5, c - 20), max(W - 0.5, c + 20))
    ax.set_ylim(max(H - 0.5, r + 20), min(-0.5, r - 20))
    ax.set_title(title)
    ax.set_xlabel("column (pixel)")
    ax.set_ylabel("row (pixel)")
    ax.legend(loc="lower left", fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("brightness temperature (K)")
    return _save(fig, path)


def plot_bt_histograms(edges_K: np.ndarray, counts_by_split: dict[str, np.ndarray], path: Path) -> Path:
    """Brightness-temperature distribution of valid core-frame pixels, one line per split (probability density)."""
    _style()
    centres = (edges_K[:-1] + edges_K[1:]) / 2
    width = float(edges_K[1] - edges_K[0])
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for (split, counts), col in zip(counts_by_split.items(), SERIES):
        total = counts.sum()
        if total == 0:
            continue
        ax.plot(centres, counts / (total * width), color=col, lw=1.5, label=f"{split} ({total / 1e6:.0f} M pixels)")
    for x, label in ((220.0, "220 K"), (180.0, "180 K")):
        ax.axvline(x, color=TEXT_2, lw=0.8, ls="--")
        ax.text(x + 1, ax.get_ylim()[1] * 0.92, label, fontsize=7, color=TEXT_2)
    ax.set_xlabel("brightness temperature (K)")
    ax.set_ylabel("probability density (1/K)")
    ax.set_xlim(170, 320)
    ax.grid(axis="y")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.46, 0.86))
    fig.tight_layout()
    return _save(fig, path)


def plot_context_availability(events: list[dict], path: Path) -> Path:
    """Fraction of windows with I_-1, I_2 and both context frames, per event."""
    _style()
    ev = [e for e in events if e.get("n_windows")]
    x = np.arange(len(ev))
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for j, (key, label, col) in enumerate((("n_ctx_before", r"$I_{-1}$", SERIES[0]), ("n_ctx_after", r"$I_2$", SERIES[1]),
                                           ("n_ctx_both", "both", SERIES[2]))):
        vals = [e[key] / e["n_windows"] for e in ev]
        ax.bar(x + (j - 1) * 0.26, vals, width=0.24, color=col, label=label)
    ax.set_xticks(x, [f"{e['event_id']}\n{e['split'][:5]}" for e in ev], fontsize=7)
    ax.set_ylabel("fraction of windows")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y")
    ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    return _save(fig, path)


def make_figures(cfg: dict) -> list[Path]:
    """The figures used in the documents: calibration check, an 11-frame window, and the coverage pass/fail pair."""
    from .dataset import load_index, read_sample

    out, made, manifest = reports_dir() / "figures", [], read_manifest()
    for event in sorted({r["event_id"] for r in manifest if r["purpose"] == "calibration_reference"}):
        l1b = {parse_abi_name(r["key"])["start_stamp"]: r for r in manifest if r["event_id"] == event and r["purpose"] == "dataset"}
        refs = [r for r in manifest if r["event_id"] == event and r["purpose"] == "calibration_reference"
                and parse_abi_name(r["key"])["start_stamp"] in l1b]
        if refs:
            scan = read_scan(from_portable(l1b[parse_abi_name(refs[0]["key"])["start_stamp"]]["local_path"]))
            ours, ref = brightness_temperature(scan["radiance"], scan["planck"]), read_cmip(from_portable(refs[0]["local_path"]))
            valid = (scan["dqf"] == 0) & (ref["dqf"] == 0) & np.isfinite(ours) & np.isfinite(ref["bt"])
            made.append(plot_calibration_difference(ours - ref["bt"], valid, ref["step"], out / f"{event}_calibration_diff.png",
                                                    title=f"{event}: our Planck inversion minus NOAA CMIP"))

    windows, patches = load_index(allow_test=True)
    both = [w for w in windows if w["ctx_before_available"] == "1" and w["ctx_after_available"] == "1"]
    if both:
        w = both[0]
        p = sorted((p for p in patches if p["window_id"] == w["window_id"]), key=lambda p: (p["kind"] != "cold_centred", p["patch_id"]))[0]
        s = read_sample(w, p, cfg["quality"]["usable_dqf"])
        made.append(plot_window_strip(s["core_bt_K"], s["t_fractions"], out / "window_example.png", s["ctx_before_bt_K"],
                                      s["ctx_after_bt_K"], (s["ctx_before_t"], s["ctx_after_t"]),
                                      title=f"{w['window_id']} / {p['kind']} patch"))

    # A storm that left the sector in one hour, beside a passing hour of the same event and sector.
    rows = read_csv(manifest_dir() / "event_verification.csv")
    coverage = {r["key"]: r for r in manifest if r["purpose"] == "coverage"}
    passed = {(r["event_id"], r["sector"]) for r in rows if r["passed"] == "True"}
    failing = [r for r in rows if r["passed"] == "False" and r["inside_with_margin"] == "False"
               and r["row"] not in ("", "nan") and (r["event_id"], r["sector"]) in passed]
    fail = failing[-1] if failing else None
    passing = [r for r in rows if fail and r["passed"] == "True" and (r["event_id"], r["sector"]) == (fail["event_id"], fail["sector"])]
    for label, r in (("fail", fail), ("pass", passing[0] if passing else None)):
        if r is None or r["key"] not in coverage:
            continue
        scan = read_scan(from_portable(coverage[r["key"]]["local_path"]))
        title = (f"{r['event_id']} {r['sector']} {r['scan_start_utc'][11:16]} UTC: margin {float(r['margin_px']):.0f} px, "
                 f"cold {100 * float(r['cold_fraction']):.1f}%")
        made.append(plot_coverage_check(brightness_temperature(scan["radiance"], scan["planck"]), (float(r["row"]), float(r["col"])),
                                        cfg["coverage"]["edge_margin_px"], out / f"coverage_{label}_{r['event_id']}.png", title))
    (out / "figures_index.json").write_text(json.dumps([str(p.relative_to(out)) for p in made], indent=2), encoding="utf-8")
    return made

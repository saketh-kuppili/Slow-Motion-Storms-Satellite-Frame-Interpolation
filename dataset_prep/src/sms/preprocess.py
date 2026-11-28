"""Phase 3: read ABI L1b scans, calibrate to brightness temperature, check quality, and store aligned segments."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
import pyproj

from .acquire import parse_abi_name, read_csv, read_manifest, write_csv
from .config import UTC, from_portable, processed_dir, reports_dir, to_portable

J2000 = dt.datetime(2000, 1, 1, 12, tzinfo=UTC)  # ABI time variables count UTC seconds from here (no leap seconds)
TIME_UNITS = "seconds since 2000-01-01 12:00:00"
DQF_NAMES = {0: "good", 1: "conditionally_usable", 2: "out_of_range", 3: "no_value",
             4: "focal_plane_temperature_threshold_exceeded"}
PROJ_KEYS = ("grid_mapping_name", "perspective_point_height", "semi_major_axis", "semi_minor_axis",
             "inverse_flattening", "latitude_of_projection_origin", "longitude_of_projection_origin", "sweep_angle_axis")
# Radiation constants from the exact SI values of h, c and k_B, in the units of ABI radiance:
# c1 = 2hc^2 in mW m^-2 sr^-1 (cm^-1)^-4, c2 = hc/k_B in cm K.
H_PLANCK, C_LIGHT, K_BOLTZMANN = 6.62607015e-34, 299_792_458.0, 1.380649e-23
C1 = 2 * H_PLANCK * C_LIGHT**2 * 1e11
C2 = H_PLANCK * C_LIGHT / K_BOLTZMANN * 100


def j2000_to_datetime(seconds: float) -> dt.datetime:
    return J2000 + dt.timedelta(seconds=float(seconds))


def datetime_to_j2000(t: dt.datetime) -> float:
    return (t - J2000).total_seconds()


# ---------------------------------------------------------------- reading L1b files
def _unsigned(var) -> np.ndarray:
    """ABI stores 12-bit counts in signed containers flagged ``_Unsigned``: reinterpret before scaling."""
    raw = np.asarray(var[:])
    if str(getattr(var, "_Unsigned", "false")).lower() == "true" and raw.dtype.kind == "i":
        raw = raw.view(raw.dtype.str.replace("i", "u"))
    return raw


def decode_packed(var) -> np.ndarray:
    """count * scale_factor + add_offset in float64; fill values and counts outside valid_range become NaN."""
    raw = _unsigned(var)
    dtype = raw.dtype
    valid = np.ones(raw.shape, bool)
    if "_FillValue" in var.ncattrs():
        valid &= raw != np.asarray(var._FillValue).astype(dtype.str.replace("u", "i")).view(dtype)
    if "valid_range" in var.ncattrs():
        lo, hi = np.asarray(var.valid_range).astype(dtype.str.replace("u", "i")).view(dtype)
        valid &= (raw >= lo) & (raw <= hi)
    out = raw.astype(np.float64) * float(getattr(var, "scale_factor", 1.0)) + float(getattr(var, "add_offset", 0.0))
    out[~valid] = np.nan
    return out


def grid_digest(shape, proj_attrs: dict, x_var, y_var) -> str:
    """Identity of the pixel grid: two scans are aligned iff projection and packed x/y coordinates are identical."""
    proj = [[k, proj_attrs[k] if isinstance(proj_attrs[k], (str, int)) else float(proj_attrs[k])]
            for k in sorted(proj_attrs)]
    coords = {name: [repr(float(v.scale_factor)), repr(float(v.add_offset)),
                     hashlib.sha1(np.ascontiguousarray(np.asarray(v[:])).tobytes()).hexdigest()]
              for name, v in (("x", x_var), ("y", y_var))}
    text = json.dumps({"shape": [int(shape[0]), int(shape[1])], "projection": proj, **coords}, sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def read_scan(path: Path) -> dict:
    """One ABI L1b radiance file: radiance (float64, NaN = no value), DQF, scan angles, times and metadata."""
    path = Path(path)
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        v = ds.variables
        proj = {}
        for k in PROJ_KEYS:
            if k in v["goes_imager_projection"].ncattrs():
                value = v["goes_imager_projection"].getncattr(k)
                proj[k] = value.item() if isinstance(value, np.generic) else value
        planck = {k: float(np.asarray(v[f"planck_{k}"][...])) for k in ("fk1", "fk2", "bc1", "bc2")}
        if -999.0 in planck.values():
            raise ValueError(f"{path.name}: no Planck coefficients (not an emissive band)")
        radiance = decode_packed(v["Rad"])
        bounds = np.asarray(v["time_bounds"][:], dtype=np.float64)
        name = parse_abi_name(path.name)
        scan = {
            "path": path, "platform": str(ds.platform_ID), "orbital_slot": str(ds.orbital_slot),
            "scene_id": str(ds.scene_id), "sector": name["scene"],
            "band_id": int(np.asarray(v["band_id"][:]).ravel()[0]), "timeline_id": str(ds.timeline_id),
            "time_start": j2000_to_datetime(bounds[0]), "time_end": j2000_to_datetime(bounds[1]),
            "time_mid": j2000_to_datetime(float(np.asarray(v["t"][...]))),
            "filename_start": name["start"],
            "radiance": radiance, "dqf": _unsigned(v["DQF"]).astype(np.uint8),
            "x": np.asarray(v["x"][:]).astype(np.float64) * float(v["x"].scale_factor) + float(v["x"].add_offset),
            "y": np.asarray(v["y"][:]).astype(np.float64) * float(v["y"].scale_factor) + float(v["y"].add_offset),
            "proj": proj, "planck": planck, "grid": grid_digest(radiance.shape, proj, v["x"], v["y"]),
            "x_image": float(np.asarray(v["x_image"][...])), "y_image": float(np.asarray(v["y_image"][...])),
            "subpoint_lon": float(np.asarray(v["nominal_satellite_subpoint_lon"][...])),
        }
    return scan


# ---------------------------------------------------------------- calibration
def brightness_temperature(radiance: np.ndarray, c: dict) -> np.ndarray:
    """T_B = (fk2 / ln(fk1/L + 1) - bc1) / bc2  (GOES-R PUG Vol. 3); NaN where L <= 0 or missing."""
    L = np.asarray(radiance, dtype=np.float64)
    out = np.full(L.shape, np.nan)
    ok = np.isfinite(L) & (L > 0)
    out[ok] = (c["fk2"] / np.log(c["fk1"] / L[ok] + 1.0) - c["bc1"]) / c["bc2"]
    return out


def radiance_from_bt(bt: np.ndarray, c: dict) -> np.ndarray:
    """Exact inverse: L = fk1 / (exp(fk2 / (bc1 + bc2 T)) - 1)."""
    return c["fk1"] / np.expm1(c["fk2"] / (c["bc1"] + c["bc2"] * np.asarray(bt, dtype=np.float64)))


def bt_sensitivity(bt: np.ndarray, c: dict) -> np.ndarray:
    """dT_B/dL, used to propagate radiance quantisation into temperature."""
    T = np.asarray(bt, dtype=np.float64)
    u = c["fk2"] / (c["bc1"] + c["bc2"] * T)
    return 1.0 / (c["fk1"] * np.exp(u) / np.expm1(u) ** 2 * u * c["bc2"] / (c["bc1"] + c["bc2"] * T))


def planck_radiance(nu_cm, T) -> np.ndarray:
    """Blackbody radiance per wavenumber, B = c1 nu^3 / (exp(c2 nu / T) - 1)."""
    nu = np.asarray(nu_cm, dtype=np.float64)
    return C1 * nu**3 / np.expm1(C2 * nu / np.asarray(T, dtype=np.float64))


def read_cmip(path: Path) -> dict:
    """NOAA's L2 CMI brightness temperature, its 12-bit quantisation step, and NOAA's own summary statistics."""
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        v = ds.variables
        summary = {k: float(v[k][...]) for k in ("min_brightness_temperature", "max_brightness_temperature",
                                                  "mean_brightness_temperature", "std_dev_brightness_temperature") if k in v}
        return {"bt": decode_packed(v["CMI"]), "dqf": np.asarray(v["DQF"][:]).view(np.uint8),
                "step": float(v["CMI"].scale_factor), "summary": summary}


def compare_with_cmip(bt: np.ndarray, dqf: np.ndarray, ref: dict, tolerance_K: float) -> dict:
    """Pixel-wise difference from NOAA on pixels good in both products.

    CMI is round((T - offset) / step), so a correct inversion differs by at most step/2
    (with a uniform rounding error, SD = step / sqrt(12)). NOAA's summary statistics are
    computed before quantisation and give a second, rounding-free check.
    """
    good = (dqf == 0) & (ref["dqf"] == 0) & np.isfinite(bt) & np.isfinite(ref["bt"])
    diff = bt[good] - ref["bt"][good]
    absd = np.abs(diff)
    ours = bt[(dqf <= 1) & np.isfinite(bt)]  # NOAA's statistics use good + conditionally usable pixels
    mine = {"min_brightness_temperature": ours.min(), "max_brightness_temperature": ours.max(),
            "mean_brightness_temperature": ours.mean(), "std_dev_brightness_temperature": ours.std()}
    summary_diffs = [abs(float(mine[k]) - val) for k, val in ref["summary"].items()]
    return {"n_pixels": int(diff.size), "mean_diff_K": float(diff.mean()), "std_diff_K": float(diff.std()),
            "max_abs_diff_K": float(absd.max()), "p99_abs_diff_K": float(np.percentile(absd, 99)),
            "quantization_step_K": ref["step"],
            "fraction_within_half_step": float(np.mean(absd <= ref["step"] / 2 + 1e-6)),
            "summary_max_abs_diff_K": float(max(summary_diffs)) if summary_diffs else float("nan"),
            "passed": bool(absd.max() <= tolerance_K)}


# ---------------------------------------------------------------- geolocation (GOES fixed grid)
def geos_projection(proj: dict) -> pyproj.Proj:
    """The ABI fixed grid as a pyproj geostationary projection (x, y in metres = scan angle * satellite height)."""
    return pyproj.Proj(proj="geos", h=proj["perspective_point_height"], lon_0=proj["longitude_of_projection_origin"],
                       sweep=proj["sweep_angle_axis"], a=proj["semi_major_axis"], b=proj["semi_minor_axis"])


def latlon_to_pixel(lat: float, lon: float, grid: dict) -> tuple[float, float]:
    """Fractional (row, col) of a location on a scan's grid (needs ``x``, ``y``, ``proj``); NaN if not visible."""
    xm, ym = geos_projection(grid["proj"])(lon, lat)
    if not (np.isfinite(xm) and np.isfinite(ym)):
        return float("nan"), float("nan")
    h = grid["proj"]["perspective_point_height"]
    x, y = grid["x"], grid["y"]
    col = (xm / h - x[0]) / ((x[-1] - x[0]) / (len(x) - 1))
    row = (ym / h - y[0]) / ((y[-1] - y[0]) / (len(y) - 1))
    return float(row), float(col)


def pixel_to_latlon(rows, cols, grid: dict) -> tuple[np.ndarray, np.ndarray]:
    h = grid["proj"]["perspective_point_height"]
    lon, lat = geos_projection(grid["proj"])(np.asarray(grid["x"])[cols] * h, np.asarray(grid["y"])[rows] * h,
                                             inverse=True)
    return np.asarray(lat), np.asarray(lon)


# ---------------------------------------------------------------- quality control and segmentation
def valid_mask(bt: np.ndarray, dqf: np.ndarray, usable_dqf=(0,)) -> np.ndarray:
    """Pixels allowed into any loss, metric or statistic: a physical value and a usable quality flag."""
    return np.isfinite(bt) & np.isin(dqf, np.asarray(usable_dqf, dtype=np.uint8))


def check_scan(scan: dict, bt: np.ndarray, satellite: str, cfg: dict) -> dict:
    """Metadata and value checks of one scan; problems are recorded, never silently fixed."""
    problems, band = [], cfg["instrument"]["bands"][0]
    expect = {"platform": f"G{satellite[-2:]}", "orbital_slot": cfg["instrument"]["orbital_slot"],
              "scene_id": "Mesoscale", "band_id": band}
    for k, want in expect.items():
        if scan[k] != want:
            problems.append(f"{k} {scan[k]!r} != {want!r}")
    if list(bt.shape) != list(cfg["quality"]["expected_shape_px"]):
        problems.append(f"shape {bt.shape}")
    lag = abs((scan["filename_start"] - scan["time_start"]).total_seconds())  # names truncate to 0.1 s
    if lag > 1.0:
        problems.append(f"file-name start differs from time_bounds by {lag:.2f} s")
    finite = bt[np.isfinite(bt)]
    lo, hi = cfg["calibration"]["physical_range_K"]
    n_out = int(np.count_nonzero((finite < lo) | (finite > hi)))
    if n_out:
        problems.append(f"{n_out} pixels outside the physical range {lo}-{hi} K")
    valid = valid_mask(bt, scan["dqf"], cfg["quality"]["usable_dqf"])
    if not valid.any():
        problems.append("no valid pixel")
    codes, counts = np.unique(scan["dqf"], return_counts=True)
    return {"key": "", "time_mid_utc": scan["time_mid"].isoformat(timespec="milliseconds"), "grid_digest": scan["grid"],
            "valid_fraction": float(valid.mean()),
            "dqf_counts": {DQF_NAMES.get(int(c), f"code_{int(c)}"): int(n) for c, n in zip(codes, counts)},
            "bt_min_K": float(finite.min()) if finite.size else float("nan"),
            "bt_max_K": float(finite.max()) if finite.size else float("nan"),
            "filename_vs_attr_start_s": lag, "problems": problems}


def segment_scans(times: list[dt.datetime], grids: list[str], max_gap_s: float, cadence_s: float) -> tuple[list[dict], list[int]]:
    """Split a scan sequence where the grid changes (sector moved) or the archive has a hole > max_gap_s.

    Shorter holes stay inside the segment (counted); windows never span them. Scans within 1 s of the
    previous one on the same grid are duplicates and dropped. Returns (segments, duplicate indices).
    """
    order = sorted(range(len(times)), key=lambda i: times[i])
    segments, duplicates, prev = [], [], None
    for i in order:
        if prev is not None:
            gap = (times[i] - times[prev]).total_seconds()
            if gap < 1.0 and grids[i] == grids[prev]:
                duplicates.append(i)
                continue
            reason = "grid_change" if grids[i] != grids[prev] else ("gap" if gap > max_gap_s else None)
            if reason:
                segments.append({"members": [], "grid": grids[i], "holes": 0, "break_reason": reason})
            elif gap > 1.5 * cadence_s:
                segments[-1]["holes"] += 1
        else:
            segments.append({"members": [], "grid": grids[i], "holes": 0, "break_reason": "start"})
        segments[-1]["members"].append(i)
        prev = i
    return segments, duplicates


def interval_statistics(times: list[dt.datetime]) -> dict:
    if len(times) < 2:
        return {"n_intervals": 0}
    d = np.diff([t.timestamp() for t in times])
    return {"n_intervals": int(d.size), "min_s": float(d.min()), "median_s": float(np.median(d)), "mean_s": float(d.mean()),
            "max_s": float(d.max()), "p01_s": float(np.percentile(d, 1)), "p99_s": float(np.percentile(d, 99))}


# ---------------------------------------------------------------- segment files (NetCDF-4)
def write_segment(path: Path, header: dict, scans) -> None:
    """One file per aligned segment: bt (float32 K, never clipped) and dqf per frame, with times and the grid.

    Frames are written one at a time (bounded memory) into separate compressed chunks (1, H, W),
    so later a window or patch reads only the frames it needs. Written to .part, then renamed.
    """
    first = header["first_scan"]
    H, W = first["radiance"].shape
    tmp = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_segment_file(tmp, header, scans, H, W)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)


def _write_segment_file(tmp: Path, header: dict, scans, H: int, W: int) -> None:
    first = header["first_scan"]
    with netCDF4.Dataset(tmp, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("y", H)
        ds.createDimension("x", W)
        attrs = {"Conventions": "CF-1.7", "title": "Slow-Motion Storms calibrated ABI segment", "format_version": "1",
                 "segment_id": header["segment_id"], "event_id": header["event_id"], "satellite": header["satellite"],
                 "sector": first["sector"], "band_id": int(header["band"]), "grid_digest": first["grid"],
                 "processing_version": header["processing_version"],
                 "created_utc": dt.datetime.now(UTC).isoformat(timespec="seconds"),
                 "orbital_slot": first["orbital_slot"], "platform_ID": first["platform"],
                 "timeline_id": first["timeline_id"], "break_reason_before": header["break_reason"],
                 "x_image": first["x_image"], "y_image": first["y_image"],
                 "nominal_satellite_subpoint_lon": first["subpoint_lon"]}
        for k, val in attrs.items():
            ds.setncattr(k, val)
        proj = ds.createVariable("goes_imager_projection", "i4")
        for k, val in first["proj"].items():
            proj.setncattr(k, val)
        for name in ("x", "y"):
            var = ds.createVariable(name, "f8", (name,))
            var.units, var.standard_name = "rad", f"projection_{name}_coordinate"
            var[:] = first[name]
        for name, long_name in (("time_mid", "scan mid-point"), ("time_start", "scan start"), ("time_end", "scan end")):
            var = ds.createVariable(name, "f8", ("time",))
            var.units, var.long_name = TIME_UNITS, long_name
        ds["time_mid"].standard_name = "time"
        bt = ds.createVariable("bt", "f4", ("time", "y", "x"), zlib=True, complevel=4, shuffle=True,
                               chunksizes=(1, H, W), fill_value=np.float32(np.nan))
        bt.units, bt.standard_name = "K", "toa_brightness_temperature"
        bt.long_name = f"ABI band {header['band']} brightness temperature (Planck inversion of L1b radiance)"
        bt.grid_mapping, bt.ancillary_variables = "goes_imager_projection", "dqf"
        dqf = ds.createVariable("dqf", "u1", ("time", "y", "x"), zlib=True, complevel=4, shuffle=True, chunksizes=(1, H, W))
        dqf.flag_values = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
        dqf.flag_meanings = ("good_pixel_qf conditionally_usable_pixel_qf out_of_range_pixel_qf "
                             "no_value_pixel_qf focal_plane_temperature_threshold_exceeded_qf")
        ds.createVariable("source_key", str, ("time",))
        for k in ("fk1", "fk2", "bc1", "bc2"):
            ds.createVariable(f"planck_{k}", "f8", ("time",))
        for i, (scan, bt_K, key) in enumerate(scans):
            if scan["grid"] != first["grid"]:
                raise RuntimeError("grid changed inside a segment")
            ds["bt"][i] = bt_K.astype(np.float32)
            ds["dqf"][i] = scan["dqf"]
            for name in ("time_mid", "time_start", "time_end"):
                ds[name][i] = datetime_to_j2000(scan[name])
            ds["source_key"][i] = key
            for k in ("fk1", "fk2", "bc1", "bc2"):
                ds[f"planck_{k}"][i] = scan["planck"][k]


def read_segment(path: Path) -> dict:
    """Segment metadata: times (J2000 s), scan angles, projection, attributes and source keys (no pixels)."""
    with netCDF4.Dataset(path) as ds:
        return {"path": Path(path), "attrs": {k: ds.getncattr(k) for k in ds.ncattrs()},
                "time_mid": np.asarray(ds["time_mid"][:], dtype=np.float64),
                "x": np.asarray(ds["x"][:]), "y": np.asarray(ds["y"][:]),
                "proj": {k: ds["goes_imager_projection"].getncattr(k) for k in ds["goes_imager_projection"].ncattrs()},
                "source_keys": [str(k) for k in ds["source_key"][:]],
                "planck_first": {k: float(ds[f"planck_{k}"][0]) for k in ("fk1", "fk2", "bc1", "bc2")}}


def open_segment(path: Path) -> netCDF4.Dataset:
    """Keep a segment open while reading many windows from it (reopening per read is slow on network drives)."""
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    return ds


def read_frames(src, frames, rows=slice(None), cols=slice(None)) -> tuple[np.ndarray, np.ndarray]:
    """Brightness temperature (K, float32) and DQF of one frame or a list of frames; src is a path or an open segment."""
    if isinstance(src, netCDF4.Dataset):
        return np.asarray(src["bt"][frames, rows, cols]), np.asarray(src["dqf"][frames, rows, cols])
    with open_segment(src) as ds:
        return read_frames(ds, frames, rows, cols)


# ---------------------------------------------------------------- Phase 3 driver (sms process)
SEGMENT_INDEX_FIELDS = ["segment_id", "event_id", "satellite", "sector", "band", "n_frames", "start_utc", "end_utc",
                        "n_holes", "break_reason_before", "grid_digest", "median_interval_s", "max_interval_s",
                        "mean_valid_fraction", "path", "size_bytes"]
SCAN_QC_FIELDS = ["key", "time_mid_utc", "grid_digest", "valid_fraction", "bt_min_K", "bt_max_K",
                  "filename_vs_attr_start_s", "dqf_counts", "problems"]


def segments_root() -> Path:
    return processed_dir() / "segments"


def read_segment_index() -> list[dict]:
    return read_csv(segments_root() / "index.csv")


def process_event(event_id: str, cfg: dict) -> dict:
    """Calibrate, quality-check and validate every downloaded scan of an event, then write its segments."""
    band = cfg["instrument"]["bands"][0]
    manifest = read_manifest()
    rows = sorted((r for r in manifest if r["event_id"] == event_id and r["purpose"] == "dataset" and r["channel"] == band),
                  key=lambda r: r["scan_start_utc"])
    if not rows:
        raise ValueError(f"no downloaded dataset files for {event_id}")
    refs = {parse_abi_name(r["key"])["start_stamp"]: r for r in manifest
            if r["event_id"] == event_id and r["purpose"] == "calibration_reference" and r["channel"] == band}

    def load(row):
        scan = read_scan(from_portable(row["local_path"]))
        return scan, brightness_temperature(scan["radiance"], scan["planck"])

    # Pass 1: check every scan and compare with NOAA where a reference was downloaded.
    qcs, keep, times, grids, calib = [], [], [], [], []
    for i, row in enumerate(rows):
        scan, bt = load(row)
        qc = check_scan(scan, bt, row["satellite"], cfg)
        qc["key"] = row["key"]
        qcs.append(qc)
        if not any(p.split()[0] in ("platform", "orbital_slot", "scene_id", "band_id", "shape") for p in qc["problems"]):
            keep.append(i)
            times.append(scan["time_mid"])
            grids.append(scan["grid"])
        stamp = parse_abi_name(row["key"])["start_stamp"]
        if stamp in refs:
            cmp = compare_with_cmip(bt, scan["dqf"], read_cmip(from_portable(refs[stamp]["local_path"])),
                                    cfg["calibration"]["max_abs_diff_vs_noaa_K"])
            fin = np.isfinite(bt)
            cmp["float32_storage_max_error_K"] = float(np.max(np.abs(bt.astype(np.float32).astype(np.float64)[fin] - bt[fin])))
            calib.append({"key": row["key"], **cmp})

    segments, duplicates = segment_scans(times, grids, cfg["segments"]["max_gap_s"], cfg["instrument"]["nominal_cadence_s"])

    # Pass 2: write one file per segment (files are re-read rather than held in memory).
    out_dir = segments_root() / event_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{event_id}_*_C{band:02d}_seg*.nc"):
        stale.unlink()
    seg_rows = []
    for n, seg in enumerate(segments):
        members = [keep[j] for j in seg["members"]]
        first, _ = load(rows[members[0]])
        seg_id = f"{event_id}_{first['sector']}_C{band:02d}_seg{n:02d}"
        path = out_dir / f"{seg_id}.nc"
        header = {"segment_id": seg_id, "event_id": event_id, "satellite": rows[members[0]]["satellite"], "band": band,
                  "processing_version": cfg["dataset"]["processing_version"], "break_reason": seg["break_reason"],
                  "first_scan": first}
        write_segment(path, header, ((*load(rows[i]), rows[i]["key"]) for i in members))
        seg_times = [times[j] for j in seg["members"]]
        stats = interval_statistics(seg_times)
        seg_rows.append({"segment_id": seg_id, "event_id": event_id, "satellite": header["satellite"],
                         "sector": first["sector"], "band": band, "n_frames": len(members),
                         "start_utc": seg_times[0].isoformat(timespec="seconds"),
                         "end_utc": seg_times[-1].isoformat(timespec="seconds"), "n_holes": seg["holes"],
                         "break_reason_before": seg["break_reason"], "grid_digest": seg["grid"],
                         "median_interval_s": round(stats.get("median_s", float("nan")), 3),
                         "max_interval_s": round(stats.get("max_s", float("nan")), 3),
                         "mean_valid_fraction": round(float(np.mean([qcs[i]["valid_fraction"] for i in members])), 6),
                         "path": to_portable(path), "size_bytes": path.stat().st_size})

    index = [r for r in read_segment_index() if r["event_id"] != event_id] + seg_rows
    write_csv(segments_root() / "index.csv", SEGMENT_INDEX_FIELDS, sorted(index, key=lambda r: (r["event_id"], r["segment_id"])))
    summary = {
        "event_id": event_id, "band": band, "n_files": len(rows), "n_scans_with_metadata_problems": len(rows) - len(keep),
        "n_duplicates_dropped": len(duplicates), "n_segments": len(segments), "segments": seg_rows,
        "interval_statistics": interval_statistics(sorted(times)),
        "scan_qc_problems": [{"key": q["key"], "problems": q["problems"]} for q in qcs if q["problems"]],
        "valid_fraction": {"min": float(min(q["valid_fraction"] for q in qcs)),
                           "mean": float(np.mean([q["valid_fraction"] for q in qcs]))},
        "bt_range_K": [float(min(q["bt_min_K"] for q in qcs)), float(max(q["bt_max_K"] for q in qcs))],
        "calibration_checks": calib, "calibration_passed": all(c["passed"] for c in calib) if calib else None,
    }
    rep = reports_dir() / "preprocess"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"{event_id}_C{band:02d}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_csv(rep / f"{event_id}_C{band:02d}_scan_qc.csv", SCAN_QC_FIELDS,
              [{**q, "valid_fraction": f"{q['valid_fraction']:.6f}", "bt_min_K": f"{q['bt_min_K']:.3f}",
                "bt_max_K": f"{q['bt_max_K']:.3f}", "filename_vs_attr_start_s": f"{q['filename_vs_attr_start_s']:.3f}",
                "dqf_counts": json.dumps(q["dqf_counts"]), "problems": " | ".join(q["problems"])} for q in qcs])
    return summary

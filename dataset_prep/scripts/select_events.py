"""Phase 2, one-time: select storm events from NOAA records and verify mesoscale coverage.

The results are committed in data/manifest/ (event_windows.csv, event_tracks.csv,
event_verification.csv, event_selection.csv), so the main pipeline only reads them.
Run from the repository root:

    python scripts/select_events.py tracks [--events E01 ...] [--include-backups]
    python scripts/select_events.py verify [--events E01 ...]
    python scripts/select_events.py rank-days --first 2025-04-08 --last 2026-11-22
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sms.acquire import (fetch, interpolate_track, list_abi_files, merge_into_manifest, read_csv,  # noqa: E402
                         read_event_tracks, read_event_windows, write_csv)
from sms.config import UTC, from_portable, load_config, load_events, manifest_dir, raw_dir  # noqa: E402


def fetch_text(url: str) -> str:
    """Download a small text record (HURDAT2, SPC reports) once into ``data/raw/records``."""
    dest = raw_dir() / "records" / url.rsplit("/", 1)[-1]
    if not dest.exists():
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
    return dest.read_text(encoding="utf-8", errors="replace")



# ---------------------------------------------------------------- storm events: windows and tracks
def parse_hurdat2(text: str) -> dict[tuple[str, int], list[dict]]:
    """HURDAT2 best tracks: header ``AL092021, IDA, 40,`` then 40 lines ``date, time, record, status, lat, lon, wind, ...``."""
    storms, lines, i = {}, text.splitlines(), 0
    while i < len(lines):
        head = [f.strip() for f in lines[i].split(",")]
        if len(head) < 3 or not head[0][:2].isalpha():
            i += 1
            continue
        n = int(head[2])
        points = []
        for line in lines[i + 1: i + 1 + n]:
            f = [c.strip() for c in line.split(",")]
            lat = float(f[4][:-1]) * (-1 if f[4][-1] == "S" else 1)
            lon = float(f[5][:-1]) * (-1 if f[5][-1] == "W" else 1)
            wind = float(f[6])
            points.append({"time": dt.datetime.strptime(f[0] + f[1], "%Y%m%d%H%M").replace(tzinfo=UTC),
                           "lat": lat, "lon": lon, "wind_kt": wind if wind >= 0 else None, "record": f[2]})
        storms[(head[1] if head[1] != "UNNAMED" else head[0], int(head[0][4:8]))] = points
        i += 1 + n
    return storms


def saffir_simpson_category(wind_kt: float) -> str:
    for threshold, cat in ((137, "5"), (113, "4"), (96, "3"), (83, "2"), (64, "1"), (34, "TS")):
        if wind_kt >= threshold:
            return cat
    return "TD"


def _wind_at(points: list[dict], when: dt.datetime) -> float:
    pts = [p for p in points if p["wind_kt"] is not None]
    return float(np.interp(when.timestamp(), [p["time"].timestamp() for p in pts], [p["wind_kt"] for p in pts]))


def sample_times(window: dict) -> list[dt.datetime]:
    """One verification time per hour of the window: the middle of each hour."""
    n = int(round((window["end"] - window["start"]).total_seconds() / 3600))
    return [window["start"] + dt.timedelta(minutes=30 + 60 * k) for k in range(n)]


def hurricane_window(ev: dict, points: list[dict], hours: float) -> tuple[dict, list[dict]]:
    """The 6 h ending at the first intensity peak inside the search interval (the intensification phase)."""
    tr = ev["track"]
    inside = [p for p in points if tr["search_start"] <= p["time"] <= tr["search_end"] and p["wind_kt"] is not None]
    if not inside:
        raise ValueError(f"{ev['id']}: no best-track point inside the search interval")
    peak_wind = max(p["wind_kt"] for p in inside)
    peak = min((p for p in inside if p["wind_kt"] == peak_wind), key=lambda p: p["time"])
    end = peak["time"]
    if end.minute or end.second:
        end = end.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
    start = end - dt.timedelta(hours=hours)
    dv24 = _wind_at(points, end) - _wind_at(points, end - dt.timedelta(hours=24))
    tags = {"peak_time": peak["time"].isoformat(), "peak_wind_kt": float(peak_wind),
            "category_at_end": saffir_simpson_category(_wind_at(points, end)),
            "category_at_start": saffir_simpson_category(_wind_at(points, start)),
            "wind_change_24h_kt": round(dv24, 1), "rapid_intensification": "yes" if dv24 >= 30 else "no"}
    window = {"event_id": ev["id"], "name": ev["name"], "type": ev["type"], "satellite": ev["satellite"],
              "start": start, "end": end, "rule": "hurdat2: 6 h ending at first intensity peak in search interval",
              "tags": tags}
    track = []
    for k in range(int(hours) + 1):  # hourly positions; the best track is 6-hourly
        t = start + dt.timedelta(hours=k)
        lat, lon = interpolate_track(points, t)
        track.append({"time": t, "lat": lat, "lon": lon, "wind_kt": round(_wind_at(points, t), 1), "n_reports": None})
    return window, track


def parse_spc_reports(text: str, day: dt.date) -> list[dict]:
    """SPC filtered reports: tornado, wind and hail tables, times UTC HHMM in the 12-to-12 UTC convective day."""
    reports, kind = [], None
    for row in csv.reader(io.StringIO(text)):
        if row and row[0] == "Time":
            kind = {"F_Scale": "tornado", "Speed": "wind", "Size": "hail"}.get(row[1] if len(row) > 1 else "")
            continue
        if kind is None or len(row) < 7:
            continue
        try:
            hhmm, lat, lon = int(row[0]), float(row[5]), float(row[6])
        except ValueError:
            continue
        d = day if hhmm >= 1200 else day + dt.timedelta(days=1)
        reports.append({"time": dt.datetime(d.year, d.month, d.day, hhmm // 100, hhmm % 100, tzinfo=UTC),
                        "kind": kind, "lat": lat, "lon": lon})
    return reports


def convective_window(ev: dict, reports: list[dict], hours: float) -> tuple[dict, list[dict]]:
    """The 6 h of the convective day with the most reports; hourly median report position as the track."""
    day = ev["track"]["convective_day"]
    day_start, span = dt.datetime(day.year, day.month, day.day, 12, tzinfo=UTC), dt.timedelta(hours=hours)
    best, best_count = None, -1
    for k in range(int(24 - hours) + 1):
        s = day_start + dt.timedelta(hours=k)
        count = sum(s <= r["time"] < s + span for r in reports)
        if count > best_count:
            best, best_count = s, count
    if best_count <= 0:
        raise ValueError(f"{ev['id']}: no storm reports on convective day {day}")
    inside = [r for r in reports if best <= r["time"] < best + span]
    tags = {"n_reports_in_window": best_count, **{f"n_{k}": sum(r["kind"] == k for r in inside)
                                                  for k in ("tornado", "wind", "hail")},
            "n_reports_day": len(reports)}
    window = {"event_id": ev["id"], "name": ev["name"], "type": ev["type"], "satellite": ev["satellite"],
              "start": best, "end": best + span,
              "rule": "spc: 6 h window with most filtered storm reports in the convective day", "tags": tags}
    track, half = [], dt.timedelta(minutes=30)
    for t in sample_times(window):
        near = [r for r in reports if t - half <= r["time"] < t + half]
        if near:
            track.append({"time": t, "lat": float(np.median([r["lat"] for r in near])),
                          "lon": float(np.median([r["lon"] for r in near])), "wind_kt": None, "n_reports": len(near)})
        else:
            track.append(None)
    known = [p for p in track if p is not None]
    for i, t in enumerate(sample_times(window)):  # hours without reports: interpolate between hours with reports
        if track[i] is None:
            lat, lon = interpolate_track(known, t)
            track[i] = {"time": t, "lat": lat, "lon": lon, "wind_kt": None, "n_reports": 0}
    return window, track


def derive_event(ev: dict, sources: dict, hours: float, cache: dict) -> tuple[dict, list[dict]]:
    if ev["track"]["source"] == "hurdat2":
        if "hurdat" not in cache:
            cache["hurdat"] = parse_hurdat2(fetch_text(sources["hurdat2_url"]))
        points = cache["hurdat"][(ev["track"]["storm_name"].upper(), int(ev["track"]["season"]))]
        return hurricane_window(ev, points, hours)
    day = ev["track"]["convective_day"]
    reports = parse_spc_reports(fetch_text(sources["spc_reports_url_template"].format(yymmdd=f"{day:%y%m%d}")), day)
    return convective_window(ev, reports, hours)


def write_event_tables(windows: dict[str, dict], tracks: dict[str, list[dict]]) -> None:
    """``event_windows.csv`` and ``event_tracks.csv``, sorted by event id."""
    write_csv(manifest_dir() / "event_windows.csv",
              ["event_id", "name", "type", "satellite", "window_start_utc", "window_end_utc", "rule", "tags"],
              [{"event_id": e, "name": w["name"], "type": w["type"], "satellite": w["satellite"],
                "window_start_utc": w["start"].isoformat(), "window_end_utc": w["end"].isoformat(), "rule": w["rule"],
                "tags": ";".join(f"{k}={v}" for k, v in w["tags"].items())} for e, w in sorted(windows.items())])
    write_csv(manifest_dir() / "event_tracks.csv", ["event_id", "time_utc", "lat", "lon", "wind_kt", "n_reports"],
              [{"event_id": e, "time_utc": p["time"].isoformat(), "lat": f"{p['lat']:.3f}", "lon": f"{p['lon']:.3f}",
                "wind_kt": "" if p["wind_kt"] is None else p["wind_kt"],
                "n_reports": "" if p["n_reports"] is None else p["n_reports"]}
               for e in sorted(windows) for p in tracks.get(e, [])])



def rank_convective_days(first: dt.date, last: dt.date, sources: dict) -> list[dict]:
    """SPC report counts per convective day, sorted by tornado reports, then all reports, then date."""
    ranks, day = [], first
    while day <= last:
        try:
            reports = parse_spc_reports(fetch_text(sources["spc_reports_url_template"].format(yymmdd=f"{day:%y%m%d}")), day)
        except requests.HTTPError:  # no report file for the day
            reports = []
        counts = {k: sum(r["kind"] == k for r in reports) for k in ("tornado", "wind", "hail")}
        ranks.append({"day": day, **counts, "total": sum(counts.values())})
        day += dt.timedelta(days=1)
    return sorted(ranks, key=lambda r: (-r["tornado"], -r["total"], r["day"]))


# ---------------------------------------------------------------- coverage verification
HOUR_FIELDS = ["event_id", "sector", "sample_time_utc", "key", "scan_start_utc", "storm_lat", "storm_lon", "row", "col",
               "margin_px", "inside_with_margin", "cold_fraction", "cold_ok", "valid_fraction", "passed"]
SELECTION_FIELDS = ["event_id", "status", "sector", "satellite", "window_start_utc", "window_end_utc", "m1_files",
                    "m1_gap_ok", "m1_hours_passed", "m2_files", "m2_gap_ok", "m2_hours_passed", "reason"]


def gap_ok_fraction(starts: list[dt.datetime], start: dt.datetime, end: dt.datetime, max_gap_s: float) -> tuple[float, float]:
    """Fraction of the window covered by scan intervals <= max_gap_s (window edges count as pseudo-scans)."""
    pts = [start] + sorted(t for t in starts if start <= t < end) + [end]
    gaps = np.diff([p.timestamp() for p in pts])
    return float(gaps[gaps <= max_gap_s].sum() / (end - start).total_seconds()), float(gaps.max())


def check_hour(event_id: str, sector: str, sample_time: dt.datetime, path: Path, key: str, track: list[dict],
               cov: dict) -> dict:
    """Is the storm inside the sector (with margin), and is there deep convection (cold cloud)?"""
    from sms.preprocess import brightness_temperature, latlon_to_pixel, read_scan, valid_mask

    scan = read_scan(path)
    bt = brightness_temperature(scan["radiance"], scan["planck"])
    valid = valid_mask(bt, scan["dqf"], (0,))
    lat, lon = interpolate_track(track, scan["time_mid"])
    row, col = latlon_to_pixel(lat, lon, scan)
    H, W = bt.shape
    margin = min(row, col, H - 1 - row, W - 1 - col) if np.isfinite(row) else float("nan")
    cold = float(np.mean(bt[valid] < cov["cold_threshold_K"])) if valid.any() else 0.0
    inside, cold_ok = bool(np.isfinite(margin) and margin >= cov["edge_margin_px"]), bool(cold >= cov["min_cold_fraction"])
    return {"event_id": event_id, "sector": sector, "sample_time_utc": sample_time.isoformat(), "key": key,
            "scan_start_utc": scan["time_start"].isoformat(timespec="seconds"),
            "storm_lat": round(lat, 3), "storm_lon": round(lon, 3), "row": round(row, 1), "col": round(col, 1),
            "margin_px": round(margin, 1), "inside_with_margin": inside, "cold_fraction": round(cold, 4),
            "cold_ok": cold_ok, "valid_fraction": round(float(valid.mean()), 4), "passed": inside and cold_ok}


def verify_event(window: dict, track: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    """Coverage checks on both sectors: one scan per hour (positions and cold cloud) plus listing continuity."""
    inst, cov = cfg["instrument"], cfg["coverage"]
    results, rows = [], []
    for sector in inst["sectors"]:
        files = list_abi_files(window["satellite"], inst["product"], window["start"], window["end"],
                               scene=sector, channel=inst["bands"][0])
        frac, longest = gap_ok_fraction([f["start"] for f in files], window["start"], window["end"],
                                        cov["max_listing_gap_s"])
        res = {"sector": sector, "n_files": len(files), "gap_ok": round(frac, 4), "max_gap_s": round(longest, 1),
               "n_expected": len(sample_times(window)), "hours": []}
        for t in sample_times(window):
            f = min(files, key=lambda f: abs((f["start"] - t).total_seconds()), default=None)
            if f is None or abs((f["start"] - t).total_seconds()) > 600:
                continue  # no scan within 10 minutes: the hour counts as failed
            row = fetch(f, "coverage", window["event_id"])
            rows.append(row)
            res["hours"].append(check_hour(window["event_id"], sector, t, from_portable(row["local_path"]),
                                           f["key"], track, cov))
        res["n_passed"] = sum(h["passed"] for h in res["hours"])
        res["passed"] = (res["n_passed"] / res["n_expected"] >= cov["min_hour_pass_fraction"]
                         and res["gap_ok"] >= cov["min_gap_ok_fraction"])
        results.append(res)
    return results, rows


def select_sector(results: list[dict]) -> tuple[str | None, str]:
    """Best passing sector (most hours passed, then continuity, then mean distance from the edge)."""
    def describe(r):
        return f"{r['sector']}: hours passed {r['n_passed']}/{r['n_expected']}, gap_ok {r['gap_ok']:.2f}"
    passing = [r for r in results if r["passed"]]
    if not passing:
        return None, "; ".join(describe(r) for r in results)
    best = max(passing, key=lambda r: (r["n_passed"] / r["n_expected"], r["gap_ok"],
                                       float(np.mean([h["margin_px"] for h in r["hours"]])) if r["hours"] else 0.0))
    return best["sector"], "selected " + describe(best)


def write_verification(results: dict[str, list[dict]], selections: dict, windows: dict) -> None:
    """Per-hour and per-event verification tables; rows of events not re-verified now are kept."""
    path_h, path_s = manifest_dir() / "event_verification.csv", manifest_dir() / "event_selection.csv"
    hours = [r for r in read_csv(path_h) if r["event_id"] not in results]
    hours += [h for res in results.values() for r in res for h in r["hours"]]
    hours.sort(key=lambda r: (r["event_id"], r["sector"], r["sample_time_utc"]))
    write_csv(path_h, HOUR_FIELDS, hours)
    sel = [r for r in read_csv(path_s) if r["event_id"] not in results]
    for e, res in results.items():
        sector, reason = selections[e]
        row = {"event_id": e, "status": "pass" if sector else "fail", "sector": sector or "",
               "satellite": windows[e]["satellite"], "window_start_utc": windows[e]["start"].isoformat(),
               "window_end_utc": windows[e]["end"].isoformat(), "reason": reason}
        for r in res:
            s = r["sector"].lower()
            row |= {f"{s}_files": r["n_files"], f"{s}_gap_ok": r["gap_ok"], f"{s}_hours_passed": f"{r['n_passed']}/{r['n_expected']}"}
        sel.append(row)
    write_csv(path_s, SELECTION_FIELDS, sorted(sel, key=lambda r: r["event_id"]))




# ---------------------------------------------------------------- command line
def event_ids(args) -> list[str]:
    if args.events:
        return list(args.events)
    return [e for e, ev in load_events()["events"].items() if args.include_backups or not ev["is_backup"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("tracks", "verify"):
        sp = sub.add_parser(name)
        sp.add_argument("--events", nargs="+", help="event ids (default: all candidates)")
        sp.add_argument("--include-backups", action="store_true")
    rk = sub.add_parser("rank-days")
    rk.add_argument("--first", required=True)
    rk.add_argument("--last", required=True)
    rk.add_argument("--top", type=int, default=15)
    rk.add_argument("--output", default="convective_day_ranking.csv")
    args = p.parse_args(argv)
    cfg, cat = load_config(), load_events()

    if args.command == "tracks":
        have = (manifest_dir() / "event_windows.csv").exists()  # keep events derived earlier
        windows = read_event_windows() if have else {}
        tracks = read_event_tracks() if have else {}
        cache = {}
        for e in event_ids(args):
            windows[e], tracks[e] = derive_event(cat["events"][e], cat["sources"], cfg["coverage"]["window_hours"], cache)
            print(f"{e}: {windows[e]['start']:%Y-%m-%d %H:%M}-{windows[e]['end']:%H:%M} UTC  {windows[e]['tags']}")
        write_event_tables(windows, tracks)
    elif args.command == "verify":
        windows, tracks = read_event_windows(), read_event_tracks()
        results, selections, rows = {}, {}, []
        for e in event_ids(args):
            results[e], new_rows = verify_event(windows[e], tracks[e], cfg)
            selections[e] = select_sector(results[e])
            rows += new_rows
            print(f"{e}: {'PASS' if selections[e][0] else 'FAIL'}  {selections[e][1]}")
        merge_into_manifest(rows)
        write_verification(results, selections, windows)
    else:
        ranks = rank_convective_days(dt.date.fromisoformat(args.first), dt.date.fromisoformat(args.last), cat["sources"])
        write_csv(manifest_dir() / args.output, ["rank", "convective_day", "n_tornado", "n_wind", "n_hail", "n_total"],
                  [{"rank": k, "convective_day": r["day"].isoformat(), "n_tornado": r["tornado"], "n_wind": r["wind"],
                    "n_hail": r["hail"], "n_total": r["total"]} for k, r in enumerate(ranks, 1)])
        for k, r in enumerate(ranks[: args.top], 1):
            print(f"{k:3d}  {r['day']}  tornado {r['tornado']:4d}  total {r['total']:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

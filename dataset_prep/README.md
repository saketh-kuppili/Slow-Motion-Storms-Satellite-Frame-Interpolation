# Slow-Motion Storms: data pipeline (Phases 1–4)

Code and documentation for the **data milestone** of a research project on temporal
interpolation of geostationary infrared imagery. Given two GOES-East ABI Band 13
observations about 10 minutes apart, later phases will reconstruct the nine real
1-minute mesoscale observations between them. This repository builds the dataset
those phases need.

1. **Phase 1, formulation**: research objectives and a machine-checked dataset specification.
2. **Phase 2, acquisition**:
   - storm windows and positions derived from NOAA records (HURDAT2, SPC storm reports);
   - mesoscale coverage verification;
   - checksummed downloads from NOAA's public archive.
3. **Phase 3, calibration and preprocessing**:
   - Planck brightness temperature validated against NOAA's own product;
   - geolocation and exact grid-alignment checks;
   - quality flags, segmentation and compressed storage.
4. **Phase 4, dataset construction**:
   - 11-frame windows built from real scan times, plus real context frames where they exist;
   - 256 × 256 patch index;
   - event-level splits with leakage checks;
   - reversible normalisation;
   - an independent validator.

Models, training and evaluation (Phase 5 onward) are deliberately **not** part of this code.

**Status.** All four phases have been run on the full dataset:

- 11 storm events, 4,313 real scans, 763 windows and 3,052 patches;
- train, validation and test splits of 5, 2 and 4 events, with one GOES-19 test storm;
- every independent validation check passed.

See [`reports/dataset/final_summary.md`](reports/dataset/final_summary.md),
[`docs/data_card.md`](docs/data_card.md) and [`docs/decisions.md`](docs/decisions.md).

## Repository layout

```
configs/
  research.yaml            research question, hypotheses, endpoints (Phase 1)
  data/base.yaml           dataset specification: every tolerance and constant, and the pilot slice
  data/events.yaml         candidate storms and backups
  data/splits.yaml         event-level train/validation/test assignment
src/sms/                   the pipeline package: five core modules + cli + plots (see "Modules")
scripts/select_events.py   one-time event selection (results committed in data/manifest/)
extensions/himawari/       exploratory Himawari-9 cadence check (not part of the dataset)
tests/                     pytest suite: one file per core module + a synthetic end-to-end build
data/
  manifest/                committed: file manifest with SHA-256, event windows, tracks, verification
  sample/                  committed: four real NOAA files (1.4 MB) used by the tests
  raw/, processed/         git-ignored: downloads, calibrated segments, dataset index
reports/                   generated reports and figures (full-dataset results)
docs/
  pdf/                     the four academic documents (compiled)
  latex/                   their LaTeX sources, shared preamble and bibliography
  decisions.md, data_card.md
  archive/                 local only (git-ignored): earlier comprehensive documents
backup/                    local only (git-ignored): the implementation before consolidation
CLAUDE.md                  project conventions, including the documentation standard
noaa_download.py, jap_download.py   the original exploratory scripts (kept unchanged)
```

## Installation

Requires Python ≥ 3.10 (tested with 3.12) and internet access to the public NOAA buckets
(no account or credentials).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # or: pip install -r requirements.txt && pip install -e .
```

With [uv](https://docs.astral.sh/uv/): `uv venv && uv pip install -e ".[dev]"`.

If the repository sits on a slow drive (for example `/mnt/c` under WSL), set
`SMS_DATA_DIR` to put bulk data elsewhere.

## Running the pipeline

```bash
pytest                                # 50 tests; uses only the committed sample, no network

sms spec                              # Phase 1: check the specification
python scripts/select_events.py tracks --include-backups   # Phase 2 (once): windows + tracks from HURDAT2 / SPC
python scripts/select_events.py verify                      #   coverage check (~45 MB: 1 file/hour/sector)
sms download --pilot                  #          pilot slices (~75 MB)
sms manifest-verify                   #          recompute every SHA-256
sms process                           # Phase 3: calibrate, QC, segment, store
sms figures                           #          figures into reports/figures/
sms build-dataset                     # Phase 4: windows, patches, splits, normalisation
sms validate-dataset                  #          independent validation report
sms summarize                         #          final dataset summary (reports/dataset/final_summary.md)
```

`make pipeline-pilot` runs the same sequence; `make help` lists every target.

**Full dataset.** Download the complete 6-hour windows of the verified events, then run
`process`, `build-dataset`, `validate-dataset` and `summarize`:

```bash
sms download --events E01 E02 E03 E04 E05 E06 E07 E08 E09 B05 G01
```

This fetches 4,313 Band 13 files, 1.65 GB raw. The calibrated segments add 2.52 GB, at
0.58 MB per frame.

G01 was selected by `python scripts/select_events.py rank-days --first 2025-04-08 --last 2026-11-22` and then
verified; the rule is decision D-014 in `docs/decisions.md`.

**Himawari (exploratory).** `python extensions/himawari/hsd_cadence.py` measures the AHI
observation cadence from real file headers. It finds 3 hidden frames per 10-minute gap,
so Himawari does not meet the dataset specification.

## Modules

Plain procedural Python: functions, dictionaries and loops. There are no classes.

| Phase | Module | What it does |
|---|---|---|
| 1 | `config.py` | paths, YAML loading, split rules, the specification check |
| 2 | `acquire.py` | ABI file names, S3 listing (`s3fs`), checksummed downloads and manifest |
| 2 | `scripts/select_events.py` | one-time event selection: HURDAT2/SPC windows and tracks, coverage verification, day ranking (results committed in `data/manifest/`) |
| 3 | `preprocess.py` | L1b decoding, Planck calibration and the NOAA CMIP check, navigation (`pyproj`), grid identity, QC, segmentation, segment files |
| 4 | `dataset.py` | windows from real scan times, context frames, patches, normalisation, leakage checks, index and sample access |
| 4 | `validate.py` | the 14 independent validation checks and the final summary |
| – | `plots.py`, `cli.py` | figures from real data; the `sms` command line |

Imports run one way, `config` → `acquire` → `preprocess` → `dataset` → `validate`. The tests mirror
the modules (`test_config.py`, `test_acquire.py`, `test_preprocess.py`, `test_dataset.py`,
`test_select_events.py`) plus `test_end_to_end.py`.

The scientific explanations behind the code (the radiometry, navigation, window rules and
leakage arguments) are in the study PDFs, not in the source comments.

## Outputs

| Path | Content | Committed |
|---|---|---|
| `data/manifest/manifest.csv` | every downloaded file: bucket, key, purpose, event, scan times, size, SHA-256 | yes |
| `data/manifest/event_*.csv` | derived windows, tracks, per-hour verification, event selection | yes |
| `data/processed/segments/` | one NetCDF-4 file per aligned segment (float32 K + DQF + times + grid) | no |
| `data/processed/dataset/` | `windows.csv`, `patches.csv`, `dataset_meta.json` (an index; patches are never copied) | no |
| `reports/` | Phase 1 spec, preprocessing reports, dataset summary and validation, figures | yes |

## Documentation

The four PDFs in `docs/pdf/` are concise study notes, 6–9 pages each, one per phase.
Each explains the science, the code and the design decisions, and ends with presentation
questions and answers. The documentation standard is recorded in [`CLAUDE.md`](CLAUDE.md).

- `01_Research_Foundations.pdf`
- `02_Satellite_Data_Acquisition.pdf`
- `03_Calibration_and_Preprocessing.pdf`
- `04_Dataset_Construction.pdf`

`make docs` rebuilds them from `docs/latex/`, which needs TeX Live with `latexmk` and `biber`.
The earlier comprehensive versions (41–58 pages) are archived locally under `docs/archive/`,
which git ignores.

## Data sources and attribution

- GOES-16/19 ABI L1b and L2 CMIP files: NOAA, via the NOAA Open Data Dissemination
  Program on AWS (`noaa-goes16`, `noaa-goes19`).
- Hurricane best tracks: NOAA National Hurricane Center, HURDAT2.
- Storm reports: NOAA Storm Prediction Center, filtered daily reports.
- Himawari-9 AHI files (extension only): JMA, via `noaa-himawari9`.

These are public data. Only the four sample files in `data/sample/` are redistributed.
No third-party source code is vendored in this repository. The fixed-grid navigation
and HSD header layout are implemented from the official documentation cited in the PDFs.
`noaa_download.py` and `jap_download.py` are the project's original exploratory scripts,
retained unchanged.

## Licence

No licence has been chosen yet. Add a `LICENSE` file before publishing if others should
be allowed to reuse the code.

# fNIRS_PFC_2025
A Python pipeline for preprocessing and analyzing functional Near-Infrared Spectroscopy (fNIRS) data collected from the prefrontal cortex using Octamon devices (Artinis Medical Systems).

---

## Overview
This package processes `.txt` fNIRS files exported from Octamon systems. It walks an input folder tree, groups recordings by task type, and runs each one through the pipeline. Steps include:

- **Channel quality control (SCI/PSP,SQI)**
  Each channel is scored with the metrics you enable (`sci`, `psp`, `sqi`). By default, SCI and PSP are computed, and a channel that fails **any** enabled metric is excluded. Short channels that fail are "kept" by default, which means they stay included in the plots. It does **not** mean they are used for short-channel regression (see options below). Filtering can be turned off to score and report only.
  Luca Pollonini, Heather Bortfeld, and John S. Oghalai, "PHOEBE: a method for real time mapping of optodes-scalp coupling in functional near-infrared spectroscopy," Biomed. Opt. Express 7, 5104-5119 (2016)
  
  Sappia MS, Hakimi N, Colier WNJM, Horschig JM. Signal quality index: an algorithm for quantitative assessment of functional near infrared spectroscopy signal quality. Biomed Opt Express. 2020 Oct 27;11(11):6732-6754. doi: 10.1364/BOE.409317. PMID: 33282521; PMCID: PMC7687963.
- **Motion artifact correction (TDDR)**
  Fishburn, F.A., Ludlum, R.S., Vaidya, C.J., & Medvedev, A.V. (2019).
  *Temporal Derivative Distribution Repair (TDDR): A motion correction method for fNIRS.*
  NeuroImage, 184, 171-179. https://doi.org/10.1016/j.neuroimage.2018.09.025
- **Short-channel regression (SCR)** for superficial noise removal
- **Band-pass filtering** using a zero-phase Butterworth 4th order filter
- **Trimming** of the recording: an initial crop at the start, then baseline and end rest are removed, and the signal begins a configurable number of seconds after the walking-start marker
- **Region-averaged hemodynamic response** across long channels (grouped by anatomical region)
- **Outputs**: RAW and z-scored data (z-score can be skipped), per-task summary sheets, diagnostic plots (per-stage and summary; can be skipped), and a processing log

**Note:** This pipeline is highly tailored to our lab's specific walking tasks and file naming conventions.

---

## Requirements
Python 3.6 or higher

All dependencies are specified in `setup.py`. To install the package and all required libraries:
```bash
pip install -e .
```

---

## Usage

### Run from the command line:
```bash
python main.py /path/to/input /path/to/output
```

### Examples
```bash
# Defaults: SCI + PSP gate channel exclusion
python main.py data/ results/

# Use all three metrics, stricter SQI threshold, only dual-task and single-task files
python main.py data/ results/ --metrics sqi sci psp --sqi-threshold 2.5 --task-filter DT ST

# Start exactly at the walking marker and write RAW output only
python main.py data/ results/ --post-walking-trim 0 --skip-zscore

# See which task types were discovered, then exit
python main.py data/ results/ --list-tasks
```

### Command-line options
| Argument | Description | Default |
|----------|-------------|---------|
| `input_dir` | Folder tree containing the recordings | — |
| `output_dir` | Directory where processed files, figures, summaries, and the log are saved | — |
| `--fs` | Sampling rate (Hz) | 50.0 |
| `--metrics` | One or more of `sqi`, `sci`, `psp`. These metrics gate channel exclusion (a channel failing any is dropped); metrics not listed are not computed | `sci psp` |
| `--sci-threshold` | SCI threshold | `DEFAULT_SCI_THRESHOLD` in `quality_control.py` |
| `--psp-threshold` | PSP threshold | `DEFAULT_PSP_THRESHOLD` in `quality_control.py` |
| `--sqi-threshold` | SQI threshold | `DEFAULT_SQI_THRESHOLD` in `quality_control.py` |
| `--no-quality-filtering` | Score channels and report, but don't drop any | off |
| `--exclude-failing-short-channels` | Also drop short channels that fail. By default they are kept, meaning they remain included in plots (not used for short-channel regression) | off |
| `--post-walking-trim` | Seconds to skip after the walking-start marker. `0` starts exactly at the marker; baseline and end rest are cut either way | 3.0 |
| `--initial-crop` | Seconds dropped from the start of every recording | 1.0 |
| `--skip-diagnostic-plots` | Skip the per-stage and summary plots | off |
| `--skip-zscore` | RAW output only | off |
| `--task-filter TASK [TASK ...]` | Only process these task types, e.g. `DT ST fTurn` | all tasks |
| `--list-tasks` | List discovered task types (with file counts) and exit | off |
| `--log-level` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | INFO |
| `--quiet`, `-q` | No console output (the log file is still written) | off |

### Output and exit codes
- A log is written to `output_dir/fnirs_processing.log`.
- When finished, the CLI prints how many recordings were processed out of the total and how many per-task summary sheets were written.
- Exit code `0` means at least one recording was processed; `2` means none were processed (or no files matched with `--list-tasks`); `1` means the input files could not be found.

import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fnirs_PFC_2025.preprocessing.average_channels import average_channels
from fnirs_PFC_2025.preprocessing.baseline_correction import baseline_subtraction
from fnirs_PFC_2025.preprocessing.butterworth_filter import butterworth_bandpass
from fnirs_PFC_2025.preprocessing.sci import scalp_coupling_index
from fnirs_PFC_2025.preprocessing.psp import peak_spectral_power
from fnirs_PFC_2025.preprocessing.short_channel_regression import scr_regression
from fnirs_PFC_2025.preprocessing.signalqualityindex import SQI
from fnirs_PFC_2025.preprocessing.tddr import tddr
from fnirs_PFC_2025.preprocessing.z_transformation import z_transformation
from fnirs_PFC_2025.processing.quality_control import (
    ChannelQuality, QualityReport,
    DEFAULT_SQI_THRESHOLD, DEFAULT_SCI_THRESHOLD, DEFAULT_PSP_THRESHOLD,
)
from fnirs_PFC_2025.viz.plots import plot_channels_separately, plot_overall_signals

logger = logging.getLogger(__name__)

# 0-based channel numbers (cap order 1R 2R 3R 4short 5L 6short 7L 8L in 1-based).
# Short channel <-> side is my best read of the layout, not checked against the
# probe docs. If SCR looks backwards, swap the two short IDs.
RIGHT_LONG_IDS = (0, 1, 2)
LEFT_LONG_IDS = (4, 6, 7)
RIGHT_SHORT_ID = 3
LEFT_SHORT_ID = 5
SHORT_CHANNEL_IDS = {RIGHT_SHORT_ID, LEFT_SHORT_ID}

_WALK = ['W1', 'WALK', 'START_WALK', 'WALKING']
_START = ['S2', 'START', 'TASK_START', 'GO']
WALK_EVENTS = {
    'DT': _WALK, 'ST': _WALK, 'LongWalk': _WALK,
    'fTurn': _START, 'Obstacle': _START, 'Navigation': _START,
    'LShape': ['W1', 'WALK', 'START_WALK', 'S3'],
}
TASK_KIND = {
    'DT': 'long_walk', 'ST': 'long_walk', 'LongWalk': 'long_walk',
    'fTurn': 'event_dependent', 'LShape': 'event_dependent',
    'Obstacle': 'event_dependent', 'Navigation': 'event_dependent',
}
MIN_EVENTS = {'fTurn': 3, 'LShape': 3, 'Obstacle': 3, 'Navigation': 2}

_OXY = ('HbO', 'O2Hb')
_DEOXY = ('HbR', 'HHb')
_HB = re.compile(r'HbO|HbR|O2Hb|HHb')
_BAD_LABELS = {'', 'NAN', 'NONE', 'NULL'}

DPF = 6.0
SEP_LONG_CM = 3.5
SEP_SHORT_CM = 1.5

# Prahl/OMLC molar extinction (cm^-1/M): nm -> (HbO2, Hb). Only the two bands this
# device uses; 757, 759 and 839 are interpolated between the 2 nm entries.
_EXT = {
    750: (518.0, 1405.24), 752: (533.2, 1515.32), 754: (548.4, 1541.76),
    756: (562.0, 1560.48), 757: (568.0, 1560.48), 758: (574.0, 1560.48),
    759: (580.0, 1554.50), 760: (586.0, 1548.52), 762: (598.0, 1508.44),
    764: (610.0, 1459.56),
    836: (1001.2, 692.64), 838: (1011.6, 692.48), 839: (1016.8, 692.42),
    840: (1022.0, 692.36), 842: (1032.4, 692.20), 844: (1042.8, 691.96),
    846: (1050.0, 691.76), 848: (1054.0, 691.52), 850: (1058.0, 691.32),
    852: (1062.0, 691.08),
}
_EXT_WL = np.array(sorted(_EXT), dtype=float)
_EXT_HBO2 = np.array([_EXT[int(w)][0] for w in _EXT_WL])
_EXT_HB = np.array([_EXT[int(w)][1] for w in _EXT_WL])


def _eps(wl):
    if not _EXT_WL[0] <= wl <= _EXT_WL[-1]:
        raise ValueError(f"{wl} nm is outside the extinction table")
    return np.interp(wl, _EXT_WL, _EXT_HBO2), np.interp(wl, _EXT_WL, _EXT_HB)


def _pair(start, end):
    return pd.DataFrame({'Sample number': [start, end], 'Event': ['BaselineStart', 'BaselineEnd']})


def _check(label, val, thr, digits):
    if np.isnan(val):
        return f"{label} could not be computed (NaN)"
    if val < thr:
        return f"{label} {val:.{digits}f} < {thr:.2f}"
    return None


class FileProcessor:
    """One recording in, RAW (and optionally ZSCORE) grand-average CSVs out.

    Quality metrics are computed on the pre-TDDR OD. post_walking_trim_seconds=0
    starts the epoch at the walking-start marker; baseline and end rest are
    always cut.
    """

    def __init__(self, fs=50.0,
                 sqi_threshold=DEFAULT_SQI_THRESHOLD,
                 sci_threshold=DEFAULT_SCI_THRESHOLD,
                 psp_threshold=DEFAULT_PSP_THRESHOLD,
                 enabled_metrics=("sci", "psp"),
                 enable_quality_filtering=True,
                 exclude_failing_short_channels=False,
                 post_walking_trim_seconds=3.0,
                 initial_crop_seconds=1.0,
                 skip_diagnostic_plots=False,
                 compute_zscore=True):
        self.fs = fs
        self.sqi_threshold = sqi_threshold
        self.sci_threshold = sci_threshold
        self.psp_threshold = psp_threshold
        self.enabled_metrics = tuple(m.lower() for m in enabled_metrics)
        bad = set(self.enabled_metrics) - {"sqi", "sci", "psp"}
        if bad:
            raise ValueError(f"Unknown quality metric(s): {sorted(bad)}")
        if post_walking_trim_seconds < 0:
            raise ValueError("post_walking_trim_seconds cannot be negative")
        self.enable_quality_filtering = enable_quality_filtering
        self.exclude_failing_short_channels = exclude_failing_short_channels
        self.post_walking_trim_seconds = post_walking_trim_seconds
        self.initial_crop_seconds = initial_crop_seconds
        self.skip_diagnostic_plots = skip_diagnostic_plots
        self.compute_zscore = compute_zscore
        self._name = ""
        self._window = None
        self._report = None

    # ---- entry point ----------------------------------------------------

    def process_file(self, file_path, output_base_dir, input_base_dir, read_file_func):
        self._name = name = os.path.basename(file_path)
        self._window = None
        self._report = None

        rel = os.path.relpath(os.path.dirname(file_path), start=input_base_dir)
        out_dir = os.path.join(output_base_dir, rel)
        os.makedirs(out_dir, exist_ok=True)

        loaded = read_file_func(file_path)
        if not loaded or not isinstance(loaded.get('data'), pd.DataFrame) or loaded['data'].empty:
            return {'success': False, 'error': 'Failed to read data from file'}

        meta = loaded.get('metadata', {})
        subject = self._extract_subject(file_path, input_base_dir, meta.get('Subject Public ID'))
        rate = meta.get('Sample Rate (Hz)')
        if rate is not None and abs(rate - self.fs) > 1e-6:
            logger.warning("%s: header says %s Hz but fs=%s", name, rate, self.fs)

        # Sample number is used as a row index everywhere below, so make it one.
        # The export starts at 4, not 0.
        df = loaded['data'].reset_index(drop=True)
        df['Sample number'] = np.arange(len(df))
        n_crop = int(self.initial_crop_seconds * self.fs)
        if 0 < n_crop < len(df):
            df = df.iloc[n_crop:].reset_index(drop=True)
            df['Sample number'] = np.arange(len(df))

        task = self.determine_task_type(name)
        events = self._get_events(df)
        if not self._task_ok(task, events):
            logger.warning("%s: %s needs more/other event markers", name, task)
            return {'success': False, 'error': 'Task validation failed', 'validation_failed': True}

        processed = self._run_pipeline(df, out_dir, events, task, subject)
        final = self._finalize(processed, out_dir, subject, task, events)
        return {
            'success': True,
            'data': final,
            'subject': subject,
            'task_type': task,
            'output_dir': out_dir,
            'quality': self._report,
        }

    # ---- filename / metadata helpers -------------------------------------

    @staticmethod
    def determine_task_type(filename):
        s = os.path.basename(filename).upper()
        if "FTURN" in s or "F_TURN" in s:
            return "fTurn"
        if "LSHAPE" in s or "L_SHAPE" in s:
            return "LShape"
        if "OBSTACLE" in s:
            return "Obstacle"
        # lookarounds instead of \b: underscore counts as a word char there
        if "NAVIGATION" in s or re.search(r'(?<![A-Z])NAV(?![A-Z])', s):
            return "Navigation"
        if re.search(r'(?<![A-Z])DT(?![A-Z])', s):
            return "DT"
        if re.search(r'(?<![A-Z])ST(?![A-Z])', s):
            return "ST"
        if "WALK" in s:
            return "LongWalk"
        return "Unknown"

    @staticmethod
    def _extract_subject(file_path, input_base_dir, header_id=None):
        # only look below the input root, so a study folder named "Subjects" doesn't match
        rel = os.path.relpath(os.path.dirname(file_path), start=input_base_dir)
        for part in rel.split(os.sep):
            low = part.lower()
            if "ohsu_turn" in low or any(t in low for t in ("subject", "subj", "sub-")):
                return part
        return str(header_id) if header_id else "Unknown"

    def _timing(self):
        # protocol lengths in seconds; 90 s variants are encoded in the filename
        s = self._name.upper()
        short = (re.search(r'TURN[_-]?(DT|ST)', s) and 'FTURN' not in s and 'F_TURN' not in s) \
            or re.search(r'(DT|ST)[_-](AC|TMB|DM)(?![A-Z])', s)
        task = 60.0 if short else 120.0
        return {'base': 20.0, 'task': task, 'tail': 10.0, 'total': 30.0 + task}

    @staticmethod
    def _get_events(df):
        if 'Event' not in df.columns:
            return pd.DataFrame(columns=['Sample number', 'Event'])
        ev = df.loc[df['Event'].notna(), ['Sample number', 'Event']].copy()
        ev['Event'] = ev['Event'].astype(str).str.strip().str.upper()
        ev = ev[~ev['Event'].isin(_BAD_LABELS)]
        ev = ev[ev['Sample number'] <= len(df)]
        return ev.drop_duplicates('Sample number').sort_values('Sample number').reset_index(drop=True)

    @staticmethod
    def _task_ok(task, events):
        if TASK_KIND.get(task) != 'event_dependent':
            return True
        if len(events) < MIN_EVENTS[task]:
            return False
        return task == 'LShape' or 'S1' in events['Event'].values

    # ---- pipeline ---------------------------------------------------------

    def _run_pipeline(self, data, out_dir, events, task, subject):
        od_cols = [c for c in data.columns if re.match(r'CH\d+_WL\d+', c)]
        groups = {}
        for c in od_cols:
            ch, wl = re.match(r'CH(\d+)_WL(\d+)', c).groups()
            groups.setdefault(f"CH{ch}", {})[wl] = c
        groups = {ch: w for ch, w in groups.items() if len(w) == 2}
        if not groups:
            raise ValueError("no channels with two wavelengths")

        conc_raw = self._to_conc(data, groups, events, task)
        self._plot_raw(conc_raw, out_dir, events, task, subject)

        excluded, report = self._score_channels(data, groups, out_dir, conc_raw)
        self._report = report

        conc = self._to_conc(self._tddr(data), groups, events, task)

        meta_cols = [c for c in ('Sample number', 'Event') if c in data.columns]
        work = pd.concat([data[meta_cols], conc], axis=1)
        if self.enable_quality_filtering and excluded:
            ids = {re.match(r'CH\d+', c).group() for c in excluded}
            work = work.drop(columns=[c for c in work.columns if c.split()[0] in ids])

        sig_cols = [c for c in work.columns if _HB.search(c)]
        if not sig_cols:
            raise ValueError("no channels left after quality filtering")

        scr = self._scr(work[sig_cols], report)
        filt = butterworth_bandpass(scr, order=4, Wn=[0.01, 0.1], fs=self.fs)
        work[sig_cols] = filt[sig_cols]
        corrected = self._baseline(work, events, task)

        if not self.skip_diagnostic_plots:
            stages = [(1, 'Post-MBLL', conc_raw), (2, 'Post-TDDR', conc), (3, 'Post-SCR', scr),
                      (4, 'Post-Filter', filt), (5, 'Post-Baseline', corrected[sig_cols])]
            for num, label, df in stages:
                self._plot_stage(df, out_dir, events, task, subject, num, label)
            self._plot_stage_summary(stages, out_dir, events, task, subject)

        return self._trim(corrected, events, task)

    def _tddr(self, data):
        od = data.filter(regex=r'^CH\d+_WL\d+$')
        out = data.copy()
        out[od.columns] = tddr(od, sample_rate=self.fs)
        return out

    # ---- OD -> concentration ----------------------------------------------

    def _to_conc(self, df, groups, events, task):
        win = self._od_baseline_window(events, task, len(df))
        conc = pd.DataFrame(index=df.index)
        for ch, wls in groups.items():
            sep = SEP_SHORT_CM if int(ch[2:]) in SHORT_CHANNEL_IDS else SEP_LONG_CM
            (wl1, col1), (wl2, col2) = wls.items()
            od1 = self._demean(df[col1].to_numpy(dtype=float), win)
            od2 = self._demean(df[col2].to_numpy(dtype=float), win)
            e1o, e1r = _eps(float(wl1))
            e2o, e2r = _eps(float(wl2))
            det = e1o * e2r - e2o * e1r
            length = DPF * sep
            conc[f"{ch} HbO"] = (e2r * od1 - e1r * od2) / (length * det) * 1e6
            conc[f"{ch} HbR"] = (-e2o * od1 + e1o * od2) / (length * det) * 1e6
        return conc

    @staticmethod
    def _demean(sig, win):
        ref = np.nanmean(sig[win[0]:win[1]]) if win else np.nan
        if not np.isfinite(ref):
            ref = np.nanmean(sig)
        return sig - ref

    def _od_baseline_window(self, ev, task, n):
        """(start, end) of the standing baseline, used to reference OD -> delta OD."""
        min_span = max(1, int(self.fs))

        def clamp(a, b):
            a = int(max(0, min(a, n - 1)))
            b = int(max(0, min(b, n)))
            return (a, b) if b - a >= min_span else None

        base = int(self._timing()['base'] * self.fs)
        if not ev.empty:
            smp, lab = ev['Sample number'], ev['Event']
            if task == 'LShape' and len(ev) >= 3:
                w = clamp(smp.iloc[1], smp.iloc[2])
                if w:
                    return w
            if TASK_KIND.get(task) == 'long_walk':
                bounds = self._long_walk_bounds(ev, task, n)
                if bounds:
                    w = clamp(bounds[0], bounds[1])
                    if w:
                        return w
            elif (lab == 'S1').any():
                s1 = smp[lab == 'S1'].iloc[0]
                for name in ('W1', 'S2'):
                    nxt = smp[(lab == name) & (smp > s1)]
                    if not nxt.empty:
                        w = clamp(s1, nxt.iloc[0])
                        if w:
                            return w
                w = clamp(s1, s1 + base)
                if w:
                    return w
        return clamp(0, base)

    # ---- event logic ------------------------------------------------------

    def _long_walk_bounds(self, ev, task, n):
        """(baseline_start, walk_start, task_end or None) for DT/ST/LongWalk.

        Marker labels aren't trusted blindly: a W1/S2 marker is only used if it
        sits within 50% of the expected protocol timing, otherwise the position
        is inferred from S1 (or S2) plus the protocol durations.
        """
        if TASK_KIND.get(task) != 'long_walk':
            return None
        smp, lab = ev['Sample number'], ev['Event']

        def first(names):
            hit = smp[lab.isin(names)]
            return int(hit.iloc[0]) if not hit.empty else None

        s1, s2, w1 = first(['S1']), first(['S2']), first(WALK_EVENTS[task])
        if s1 is None and s2 is None and w1 is None:
            return None

        t = self._timing()
        fs = self.fs
        base_n, task_n = int(t['base'] * fs), int(t['task'] * fs)

        def close(a, b, ref):
            return abs(a - b) / fs <= 0.5 * max(ref, 1.0)

        if s1 is not None:
            guess = s1 + base_n
        elif s2 is not None:
            guess = s2 - task_n
        else:
            guess = w1

        start = guess
        for marker in (w1, s2):
            if marker is not None and close(marker, guess, t['base']):
                start = marker
                break
        if w1 is not None and w1 != start:
            logger.warning("%s: W1 marker at sample %d doesn't fit the protocol timing; using %d",
                           self._name, w1, start)

        end = None
        if s2 is not None and s2 > start and close(s2, start + task_n, t['task']):
            end = s2
        base_start = s1 if s1 is not None else start - base_n
        return base_start, start, end

    def _find_walk_start(self, ev, task):
        names = WALK_EVENTS.get(task)
        if names is None:
            return None
        smp, lab = ev['Sample number'], ev['Event']

        for name in names:
            hit = smp[lab == name]
            if not hit.empty:
                return int(hit.iloc[0])

        def after(s):
            nxt = smp[smp > s]
            return int(nxt.iloc[0]) if not nxt.empty else None

        s1 = smp[lab == 'S1']
        s1 = s1.iloc[0] if not s1.empty else None
        kind = TASK_KIND.get(task)
        if kind == 'long_walk':
            s_any = smp[lab.str.match(r'S[1-9]')]
            if len(s_any) >= 2:
                return int(s_any.iloc[1])
            if s1 is not None and after(s1) is not None:
                return after(s1)
        elif kind == 'event_dependent' and s1 is not None:
            s2 = smp[(lab == 'S2') & (smp > s1)]
            if not s2.empty:
                return int(s2.iloc[0])
            if after(s1) is not None:
                return after(s1)

        if task == 'LShape' and (lab == 'S2').any():
            s2 = smp[lab == 'S2'].iloc[0]
            w1 = smp[(lab == 'W1') & (smp > s2)]
            if not w1.empty:
                return int(w1.iloc[0])
            if after(s2) is not None:
                return after(s2)

        if len(ev) >= 2:
            return int(smp.iloc[1])
        return None

    # ---- quality ------------------------------------------------------------

    def _score_channels(self, data, groups, out_dir, conc):
        metrics = self.enabled_metrics
        report = QualityReport(metrics_used=metrics, sqi_threshold=self.sqi_threshold,
                               sci_threshold=self.sci_threshold, psp_threshold=self.psp_threshold)
        excluded = []

        for ch, wls in groups.items():
            num = int(ch[2:])
            short = num in SHORT_CHANNEL_IDS
            cols = list(wls.values())
            od1 = data[cols[0]].to_numpy(dtype=float)
            od2 = data[cols[1]].to_numpy(dtype=float)
            sqi = sci = psp = None
            why = []

            if 'sqi' in metrics:
                oxy = conc[f"{ch} HbO"].to_numpy()
                deoxy = conc[f"{ch} HbR"].to_numpy()
                sqi = float(SQI(od1, od2, oxy, deoxy, self.fs))
                why.append(_check('SQI', sqi, self.sqi_threshold, 2))
            if 'sci' in metrics:
                sci = scalp_coupling_index(od1, od2, self.fs)
                why.append(_check('SCI', sci, self.sci_threshold, 3))
            if 'psp' in metrics:
                psp = peak_spectral_power(od1, od2, self.fs)
                why.append(_check('PSP', psp, self.psp_threshold, 3))
            why = [w for w in why if w]

            # short channels are only regressors, so keep them unless told otherwise
            spared = bool(why) and short and not self.exclude_failing_short_channels
            if spared:
                why = ["short channel kept despite: " + "; ".join(why)]
            passed = spared or not why
            report.channels.append(ChannelQuality(num, short, passed=passed, sqi=sqi, sci=sci,
                                                  psp=psp, reasons=tuple(why)))

            if why and not spared and self.enable_quality_filtering:
                logger.warning("%s: excluding %s (%s)", self._name, ch, "; ".join(why))
                excluded.extend(cols)

        suffix = "_filtered" if self.enable_quality_filtering else "_unfiltered"
        stem = os.path.splitext(self._name)[0]
        report.to_dataframe().to_csv(os.path.join(out_dir, f"{stem}_quality_report{suffix}.csv"), index=False)
        return excluded, report

    def _scr(self, df, report):
        """Short-channel regression, each side with its own short channel.

        If one short channel failed QC the other side's is used for both; if
        both failed, SCR is skipped. Either case is written to report.scr_note.
        """
        def cols(ids):
            wanted = {f"CH{i}" for i in ids}
            return [c for c in df.columns if c.split()[0] in wanted]

        right_long, left_long = cols(RIGHT_LONG_IDS), cols(LEFT_LONG_IDS)
        right_short, left_short = cols([RIGHT_SHORT_ID]), cols([LEFT_SHORT_ID])
        right_ok = bool(right_short) and self._short_passed(report, RIGHT_SHORT_ID)
        left_ok = bool(left_short) and self._short_passed(report, LEFT_SHORT_ID)

        if not (right_ok or left_ok):
            report.scr_note = (f"SCR skipped: neither short channel (CH{RIGHT_SHORT_ID}, CH{LEFT_SHORT_ID}) "
                               f"genuinely passed quality; long channels left uncorrected.")
            return df

        out = df.copy()
        sides = ((right_long, right_ok, right_short, left_short),
                 (left_long, left_ok, left_short, right_short))
        for long_cols, own_ok, own, other in sides:
            short_cols = own if own_ok else other
            if long_cols and short_cols:
                out[long_cols] = scr_regression(df[long_cols], df[short_cols])[long_cols]

        if right_ok and not left_ok:
            report.scr_note = (f"SCR: left short channel (CH{LEFT_SHORT_ID}) failed quality; "
                               f"used right short channel (CH{RIGHT_SHORT_ID}) for both hemispheres.")
        elif left_ok and not right_ok:
            report.scr_note = (f"SCR: right short channel (CH{RIGHT_SHORT_ID}) failed quality; "
                               f"used left short channel (CH{LEFT_SHORT_ID}) for both hemispheres.")
        return out

    @staticmethod
    def _short_passed(report, channel):
        # a short channel "spared" despite failing has passed=True but non-empty reasons
        return any(c.channel == channel and c.passed and not c.reasons for c in report.channels)

    # ---- baseline / trimming --------------------------------------------------

    def _baseline(self, df, ev, task):
        if ev.empty:
            return self._baseline_fallback(df)

        smp, lab = ev['Sample number'], ev['Event']
        if task == 'LShape':
            if len(ev) < 3 or smp.iloc[2] <= smp.iloc[1]:
                raise ValueError("L-Shape baseline needs 3 ordered event markers")
            return baseline_subtraction(df, _pair(smp.iloc[1], smp.iloc[2]))

        if TASK_KIND.get(task) == 'long_walk':
            bounds = self._long_walk_bounds(ev, task, len(df))
            if bounds and bounds[1] > bounds[0]:
                return baseline_subtraction(df, _pair(bounds[0], bounds[1]))
            return self._baseline_fallback(df)

        if (lab == 'S1').any():
            s1 = smp[lab == 'S1'].iloc[0]
            for name in ('W1', 'S2'):
                nxt = smp[(lab == name) & (smp > s1)]
                if not nxt.empty:
                    return baseline_subtraction(df, _pair(s1, nxt.iloc[0]))
        raise ValueError("no usable baseline markers (S1 followed by W1/S2)")

    def _baseline_fallback(self, df):
        # no markers: work backwards from the end of the recording using protocol lengths
        t, fs, total = self._timing(), self.fs, len(df)
        if total < int(t['base'] * fs):
            return df

        def clip(x):
            return max(0, min(total - 1, int(x)))

        marks = sorted({clip(total - t['total'] * fs),
                        clip(total - (t['total'] - t['base']) * fs),
                        clip(total - t['tail'] * fs)})
        if len(marks) < 3 or marks[1] - marks[0] < int(2 * fs):
            start = max(0, total - int(t['total'] * fs))
            marks = [start, start + int(2 * fs)]
        logger.warning("%s: no baseline markers, using timing from end of recording", self._name)
        return baseline_subtraction(df, _pair(marks[0], marks[1]))

    def _trim(self, df, ev, task):
        """Keep only the walking epoch: walk start + trim -> task end."""
        self._window = None
        if ev.empty:
            return self._trim_fallback(df, task)

        n = len(df)
        end = None
        bounds = self._long_walk_bounds(ev, task, n)
        if bounds:
            _, walk_start, end = bounds
        else:
            walk_start = self._find_walk_start(ev, task)
        if walk_start is None:
            return self._trim_fallback(df, task)

        start = walk_start + int(round(self.post_walking_trim_seconds * self.fs))
        if start >= n:
            return self._trim_fallback(df, task)

        # NOTE: this range starts AT walk_start, so a walking-start marker counts as
        # a "critical event" and pulls the cut to marker + 1. The post-walking trim
        # therefore only bites when the start was inferred, not read from a marker.
        # Use `>` instead of `>=` to make it always apply.
        smp, lab = ev['Sample number'], ev['Event']
        hits = smp[(smp >= walk_start) & (smp < start) & lab.isin(WALK_EVENTS.get(task, []))]
        if not hits.empty:
            start = int(hits.max()) + 1

        stop = n
        if end is not None:
            if end > start:
                stop = min(end, n)
        elif TASK_KIND.get(task) == 'long_walk':
            candidate = walk_start + int(self._timing()['task'] * self.fs)
            if candidate > start:
                stop = min(candidate, n)
        return self._cut(df, start, stop)

    def _trim_fallback(self, df, task):
        t, fs, total = self._timing(), self.fs, len(df)
        if total < int(t['total'] * fs):
            logger.warning("%s: recording shorter than expected, not trimming", self._name)
            return df

        walk_start = int(total - (t['total'] - t['base']) * fs)
        start = max(0, walk_start + int(round(self.post_walking_trim_seconds * fs)))
        stop = total
        if TASK_KIND.get(task) == 'long_walk':
            stop = max(start, int(total - t['tail'] * fs))
        if start >= stop:
            return df
        logger.warning("%s: no walking-start marker, trimmed by timing from end of recording", self._name)
        return self._cut(df, start, stop)

    def _cut(self, df, start, stop):
        out = df.iloc[start:stop].copy()
        out['Sample number'] = np.arange(len(out))
        self._window = (start, stop)
        return out

    # ---- outputs ----------------------------------------------------------------

    def _finalize(self, df, out_dir, subject, task, events):
        kw = dict(short_ids=sorted(SHORT_CHANNEL_IDS), left_ids=list(LEFT_LONG_IDS),
                  right_ids=list(RIGHT_LONG_IDS))
        outputs = {'RAW': average_channels(df, **kw)}
        if self.compute_zscore:
            sig_cols = [c for c in df.columns if _HB.search(c)]
            outputs['ZSCORE'] = average_channels(z_transformation(df, sig_cols), **kw)

        tag = '_'.join(m.upper() for m in self.enabled_metrics) or 'NONE'
        suffix = f"_with_{tag}_filtering" if self.enable_quality_filtering else "_without_quality_filtering"
        folder = os.path.join(out_dir, f"{task}{suffix}")
        os.makedirs(folder, exist_ok=True)

        for kind, out in outputs.items():
            out['Time (s)'] = out['Sample number'] / self.fs
            out['Condition'] = os.path.splitext(self._name)[0]
            out['Subject'] = subject
            out['TaskType'] = task
            out['TaskCategory'] = TASK_KIND.get(task, 'unknown')
            out['Quality_Filtering_Applied'] = bool(self.enable_quality_filtering)
            out['Quality_Metrics_Used'] = '+'.join(self.enabled_metrics) or 'none'
            out.to_csv(os.path.join(folder, f"{self._name}_FULLY_PROCESSED_{kind}{suffix}.csv"), index=False)

        titles = {'RAW': 'Raw Concentrations', 'ZSCORE': 'Z-scores'}
        try:
            for kind, out in outputs.items():
                self._plot_final(out, folder, f"{task}{suffix}", f"final_overall_{kind}{suffix}",
                                 f"Final Overall - {titles[kind]}{suffix}", events, subject)
        except Exception:
            logger.warning("%s: final plots failed", self._name, exc_info=True)
        return outputs['RAW']

    # ---- plotting -----------------------------------------------------------------

    @staticmethod
    def _save(fig, path):
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _is_short(col):
        m = re.match(r'CH(\d+)', col)
        return bool(m) and int(m.group(1)) in SHORT_CHANNEL_IDS

    def _long_cols(self, cols, keys):
        return [c for c in cols if any(k in c for k in keys) and not self._is_short(c)]

    def _grand(self, df):
        oxy, deoxy = self._long_cols(df.columns, _OXY), self._long_cols(df.columns, _DEOXY)
        if not oxy or not deoxy:
            return None
        return pd.DataFrame({'Time (s)': np.arange(len(df)) / self.fs,
                             'grand oxy': df[oxy].mean(axis=1),
                             'grand deoxy': df[deoxy].mean(axis=1)})

    def _plot_raw(self, conc, out_dir, events, task, subject):
        oxy, deoxy = self._long_cols(conc.columns, _OXY), self._long_cols(conc.columns, _DEOXY)
        if not oxy or not deoxy:
            return
        folder = os.path.join(out_dir, task)
        os.makedirs(folder, exist_ok=True)
        title = f"{self._name} - Raw Concentration"
        fig = plot_channels_separately(conc[oxy + deoxy], fs=self.fs, subject=subject, condition=task,
                                       title=f"{title} (Post-MBLL, Pre-Processing)")
        self._save(fig, os.path.join(folder, f"raw_concentration_individual_channels_{task}.png"))

        fig = plot_overall_signals(self._grand(conc), fs=self.fs, subject=subject, condition=task,
                                   title=f"{title} Overall (Post-MBLL, Pre-Processing)",
                                   events=events[events['Sample number'] < len(conc)])
        self._save(fig, os.path.join(folder, f"raw_concentration_overall_{task}.png"))

    def _plot_stage(self, df, out_dir, events, task, subject, num, label):
        grand = self._grand(df)
        if grand is None:
            return
        folder = os.path.join(out_dir, task, "diagnostic_stages")
        os.makedirs(folder, exist_ok=True)
        fig = plot_overall_signals(grand, fs=self.fs, subject=subject, condition=task,
                                   title=f"{self._name} - Stage {num}: {label}",
                                   events=events[events['Sample number'] < len(df)])
        fname = f"stage_{num}_{label.replace(' ', '_').replace('-', '_')}_{task}.png"
        self._save(fig, os.path.join(folder, fname))

    def _plot_stage_summary(self, stages, out_dir, events, task, subject):
        folder = os.path.join(out_dir, task, "diagnostic_stages")
        os.makedirs(folder, exist_ok=True)
        fig, axes = plt.subplots(len(stages), 1, figsize=(14, 4 * len(stages)), sharex=True)
        fig.suptitle(f"{self._name} - Processing Pipeline Stages\nSubject: {subject}\n"
                     f"(Long channels only, excluding CH3 & CH5)", fontsize=12)
        n_max = max(len(df) for _, _, df in stages)
        ev = events[events['Sample number'] < n_max]

        for ax, (num, label, df) in zip(axes, stages):
            grand = self._grand(df)
            if grand is None:
                ax.text(0.5, 0.5, f"No long-channel data for {label}", ha='center', va='center')
                ax.set_title(f"{num}_{label}")
                continue
            time = grand['Time (s)']
            oxy, deoxy = grand['grand oxy'], grand['grand deoxy']
            ax.plot(time, oxy, 'r-', label='HbO', linewidth=1.2)
            ax.plot(time, deoxy, 'b-', label='HbR', linewidth=1.2)

            lo, hi = ax.get_ylim()
            for _, row in ev.iterrows():
                t = row['Sample number'] / self.fs
                if t <= time.iloc[-1]:
                    ax.axvline(x=t, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
                    ax.text(t, hi - 0.05 * (hi - lo), row['Event'], rotation=90, va='top',
                            ha='right', fontsize=7, alpha=0.8)
            ax.set_title(f"{num}_{label} (HbO: {oxy.mean():.2f}+/-{oxy.std():.2f}, "
                         f"HbR: {deoxy.mean():.2f}+/-{deoxy.std():.2f} uM)")
            ax.set_ylabel("Delta[Hb] (uM)")
            ax.legend(loc='upper right', fontsize=8)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save(fig, os.path.join(folder, f"SUMMARY_all_stages_{task}.png"))

    def _plot_final(self, df, folder, tag, prefix, title, events, subject):
        if self._window:
            start, stop = self._window
            ev = events[(events['Sample number'] >= start) & (events['Sample number'] < stop)].copy()
            ev['Sample number'] -= start
        else:
            ev = events[events['Sample number'] < len(df)]
        fig = plot_overall_signals(df[['grand oxy', 'grand deoxy', 'Time (s)']], fs=self.fs,
                                   title=f"{self._name} - {title}", subject=subject,
                                   condition=tag, events=ev)
        self._save(fig, os.path.join(folder, f"{prefix}_{tag}.png"))

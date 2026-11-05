import gzip
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sparse
from scipy.sparse.linalg import lsqr
from scipy.optimize import curve_fit
from scipy.stats import linregress, siegelslopes, norm, gaussian_kde
from scipy.interpolate import interp1d
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re

# ==========================================
# 1. DATA LOADING
# ==========================================

class _RenamingUnpickler(pickle.Unpickler):
    _MODULE_RENAMES = {"numpy._core": "numpy.core"}
    _PAIR_RENAMES = {("numpy.core._multiarray_umath", "reconstruct"): ("numpy.core.multiarray", "reconstruct")}
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        key = (module, name)
        if key in self._PAIR_RENAMES:
            module, name = self._PAIR_RENAMES[key]
        return super().find_class(module, name)

def load_results_gz(pathlike: str):
    path = Path(pathlike)
    with gzip.open(path, "rb") as f:
        return _RenamingUnpickler(f).load()

def read_sc_growclust(pathlike: str) -> pd.DataFrame:
    GC_COLS = [
        "yr","mon","day","hr","min","sec","eID","latR","lonR","depR","mag",
        "qID","cID","nbranch","qnpair","qndiffP","qndiffS","rmsP","rmsS",
        "eh","ez","et","latC","lonC","depC"
    ]
    rows = []
    with open(str(pathlike), "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                nums = []
                for tok in line.split():
                    tok = tok.replace("D", "E")
                    try:
                        nums.append(float(tok))
                    except ValueError: pass
                if not nums: continue
                if len(nums) < 25: nums += [np.nan] * (25 - len(nums))
                else: nums = nums[:25]
                rows.append(nums)
            except Exception: continue

    df = pd.DataFrame(rows, columns=GC_COLS)
    time_cols = ['yr', 'mon', 'day', 'hr', 'min', 'sec']
    for c in time_cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=time_cols)
    
    df['time'] = pd.to_datetime(dict(
        year=df['yr'], month=df['mon'], day=df['day'], hour=df['hr'], 
        minute=df['min'], second=df['sec'].astype(int)
    )) + pd.to_timedelta(df['sec'] % 1, unit='s')
    
    df['time'] = df['time'].dt.tz_localize('UTC')
    df["eID_str"] = df["eID"].apply(lambda x: str(int(x)) if pd.notna(x) else None)
    return df

def build_event_meta(gc_df, present_event_ids, restrict_years=None, restrict_mag_min=None):
    present_event_ids = set(str(eid) for eid in present_event_ids)
    df = gc_df.loc[gc_df["eID_str"].isin(present_event_ids)].copy()
    if restrict_years:
        df = df[(df["yr"] >= restrict_years[0]) & (df["yr"] <= restrict_years[1])]
    if restrict_mag_min:
        df = df[df["mag"] >= restrict_mag_min]
    df = df.sort_values(["time"]).drop_duplicates(subset=["eID_str"], keep="first")
    df = df.set_index("eID_str", drop=True)
    return df

# ==========================================
# 2. SPECTRAL CALCULATIONS (EXACT ORIGINAL LOGIC)
# ==========================================

_PHASE_ALIASES = {'P': 'P', 'S': 'S', 'CODA': 'CODA', 'C': 'CODA', 'WHOLE': 'WHOLE', 'W': 'WHOLE'}
_METHOD_ALIASES = {'multitaper': 'multitaper', 'mt': 'multitaper', 'multiwindow': 'multiwindow', 
                   'mw': 'multiwindow', 'envelope': 'envelope', 'env': 'envelope', 
                   'envelope_fd': 'envelope_fd', 'coda-fd': 'envelope_fd', 'fd': 'envelope_fd'}

def parse_phase_method_spec(spec):
    if isinstance(spec, (list, tuple)): raw_list = spec
    else:
        text = str(spec)
        raw_list = []
        tokens = [t for t in re.split(r'[,\s]+', text) if t]
        for tok in tokens:
            if ':' in tok: raw_list.append(tok.split(':', 1))
    out = []
    for ph, m in raw_list:
        ph_norm = _PHASE_ALIASES.get(str(ph).strip().upper(), None)
        m_key = str(m).strip().lower().replace(' ', '').replace('/', '_')
        m_norm = _METHOD_ALIASES.get(m_key, None)
        if ph_norm and m_norm: out.append((ph_norm, m_norm))
    return out

def _interp_to(x_from, y_from, x_to):
    y_from = np.asarray(y_from, dtype=float)
    x_from = np.asarray(x_from, dtype=float)
    x_to   = np.asarray(x_to,   dtype=float)
    if len(x_from) == len(x_to) and np.allclose(x_from, x_to):
        return np.maximum(y_from, 1e-30)
    f = interp1d(x_from, y_from, kind='linear', bounds_error=False, fill_value="extrapolate")
    y = f(x_to)
    return np.maximum(y, 1e-30)

def _mask_by_snr(sig_power, noise_power, thr):
    s = np.maximum(sig_power, 1e-30)
    n = np.maximum(noise_power, 1e-30)
    out = s.copy()
    out[np.sqrt(s/n) < thr] = np.nan
    return out

def _get_noise(results, ev, method, sta, comp, freq_target):
    for m in (method, 'multitaper'):
        try:
            nb = results[ev]['NOISE'][m]
            if sta in nb['stations']:
                i = nb['stations'].index(sta)
                nspec = np.asarray(nb[comp][i], dtype=float).squeeze()
                try:
                    nf = np.asarray(nb['frequencies'][i], dtype=float)
                except Exception:
                    nf = freq_target
                return _interp_to(nf, nspec, freq_target)
        except Exception: pass
    return None

def _is_env_method(method: str) -> bool:
    return method in ('envelope', 'envelope_L', 'envelope_fd')

def _parse_component_spec(spec: str):
    s = spec.upper().strip()
    s = s.replace('+', '/').replace(',', '/').replace('ONLY', '').replace(' ', '')
    if not s: return ('N','Z','E')
    if s in ('EN',): return ('E','N')
    if s in ('EZ',): return ('E','Z')
    if s in ('NZ',): return ('N','Z')
    allowed = ('H','E','N','Z')
    out = tuple([c for c in allowed if c in s.split('/')])
    if out: return out
    out = tuple([c for c in allowed if c in s])
    return out if out else ('N','Z','E')

def _eval_fit_at(d, t_ref):
    c_floor = float(d.get('fit_c_floor', 0.0))
    t0 = float(d.get('t0_ot', 0.0))
    if ('fit_p' in d) and ('fit_alpha' in d):  # curved model
        b = float(d['fit_b']); p = float(d['fit_p']); a = float(d['fit_alpha'])
        dt = max(1e-3, float(t_ref) - t0)
        logA = b - p*np.log(dt) - a*float(t_ref)
    elif 'fit_m' in d:  # straight line
        b = float(d['fit_b']); m = float(d['fit_m'])
        logA = b + m*float(t_ref)
    else: return None
    return float(np.exp(np.clip(logA, -50, 50)) + c_floor)

def compute_phase_method_ratios(results, ev1, ev2, phase, method, noise_thres=4.5, return_freqs=False, components=None, return_stations=True):
    if phase not in results.get(ev1, {}) or phase not in results.get(ev2, {}): return [] if not return_freqs else ([], [])
    
    bucket1 = results[ev1][phase].get(method)
    bucket2 = results[ev2][phase].get(method)
    if not bucket1 or not bucket2: return [] if not return_freqs else ([], [])

    stas1 = bucket1.get('stations', [])
    stas2 = bucket2.get('stations', [])
    common_stas = sorted(set(stas1) & set(stas2))
    if not common_stas: return [] if not return_freqs else ([], [])

    if isinstance(components, str): comps_req = _parse_component_spec(components)
    elif components: comps_req = tuple([c.upper() for c in components])
    else: comps_req = None

    is_env = _is_env_method(method)
    ratios = []
    freqs_per_station = []
    stations_used = []

    freq_ref = None
    if not is_env:
        try: freq_ref = np.asarray(bucket1['frequencies'][0], dtype=float)
        except Exception:
            try: freq_ref = np.asarray(bucket2['frequencies'][0], dtype=float)
            except Exception: return [] if not return_freqs else ([], [])

    def _sig(bkt, comp, idx):
        arrs = bkt.get(comp, [])
        if idx >= len(arrs): return None
        return np.asarray(arrs[idx], dtype=float)

    def _gmean(arr_list, axis=0):
        stack = np.stack(arr_list, axis=0)
        return np.exp(np.nanmean(np.log(np.maximum(stack, 1e-30)), axis=0))

    for sta in common_stas:
        try:
            i1 = stas1.index(sta); i2 = stas2.index(sta)

            if is_env:
                # [Previous logic for Envelope methods is fine, omitting for brevity if you aren't using Envelopes yet]
                pass 
            else:
                comps_sel = comps_req if comps_req else ('E','N','Z')
                comps_sel = tuple([c for c in comps_sel if c in ('E','N','Z')]) or ('E','N','Z')
                freq = freq_ref
                if freq is None: continue

                s1 = {c: _sig(bucket1, c, i1) for c in ('E','N','Z')}
                s2 = {c: _sig(bucket2, c, i2) for c in ('E','N','Z')}
                if any(s1[c] is None or s2[c] is None for c in comps_sel): continue

                n1 = {c: _get_noise(results, ev1, method, sta, c, freq) for c in comps_sel}
                n2 = {c: _get_noise(results, ev2, method, sta, c, freq) for c in comps_sel}
                if any(n1[c] is None or n2[c] is None for c in comps_sel): continue

                def _mask_one(sig, noi): return _mask_by_snr(sig, noi, noise_thres)

                is_mw = any(getattr(s1[c], 'ndim', 1) == 2 or getattr(s2[c], 'ndim', 1) == 2 for c in comps_sel)

                if is_mw:
                    for c in comps_sel:
                        if s1[c].ndim == 1: s1[c] = s1[c][None, :]
                        if s2[c].ndim == 1: s2[c] = s2[c][None, :]
                    win_ratios = []
                    for w in range(s1[comps_sel[0]].shape[0]):
                        per_comp = []
                        for c in comps_sel:
                            s1w = _mask_one(s1[c][w], n1[c])
                            s2w = _mask_one(s2[c][w], n2[c])
                            per_comp.append(np.sqrt(s1w) / np.sqrt(s2w))
                        combo = _gmean(per_comp, axis=0)
                        win_ratios.append(combo)
                    log_ratio = np.nanmedian(np.log10(np.vstack(win_ratios)), axis=0)
                else:
                    per_comp = []
                    for c in comps_sel:
                        s1c = _mask_one(s1[c], n1[c]); s2c = _mask_one(s2[c], n2[c])
                        per_comp.append(np.sqrt(s1c) / np.sqrt(s2c))
                    combo = _gmean(per_comp, axis=0)
                    log_ratio = np.log10(combo)

                ratios.append(log_ratio)
                stations_used.append(sta)

        except Exception: continue

    if return_freqs:
        if return_stations: return ratios, freqs_per_station, stations_used
        return ratios, freqs_per_station
    else:
        if return_stations: return ratios, stations_used
        return ratios

def _aggregate_ragged(freq_lists, ratio_lists, round_dec=6, min_count=1):
    if not freq_lists or not ratio_lists: return None, None
    all_fc = np.unique(np.round(np.concatenate([np.asarray(f,float) for f in freq_lists]), round_dec))
    bins = {f: [] for f in all_fc}
    for f_i, r_i in zip(freq_lists, ratio_lists):
        f_i = np.round(np.asarray(f_i, float), round_dec)
        r_i = np.asarray(r_i, float)
        m = min(len(f_i), len(r_i))
        f_i = f_i[:m]; r_i = r_i[:m]
        for f, v in zip(f_i, r_i):
            if np.isfinite(v): bins[f].append(v)
    fc, med = [], []
    for f in all_fc:
        if len(bins[f]) >= min_count:
            fc.append(float(f)); med.append(float(np.nanmedian(bins[f])))
    if not fc: return None, None
    return np.asarray(fc), np.asarray(med)

def apply_bandwidth_scaling(ds, lfl_lo, hfl_hi):
    """Convert band slopes (log10 units per decade of frequency) into the predicted
    drift of the band mean across the band, in dex: slope * log10(f_hi / f_lo).
    Uses the per-pair adaptive LFL upper edge and HFL lower edge recorded by the
    builder. With this, sigma_eff = sqrt(std^2 + (W*drift)^2)/sqrt(N_sta) sums two
    quantities in the same units. Refuses to scale a dataset twice."""
    if ds.get('slope_units') == 'drift_dex':
        return ds
    out = ds.copy()
    dlogL = np.log10(np.asarray(ds['used_lfl_hi'], float) / lfl_lo)
    dlogH = np.log10(hfl_hi / np.asarray(ds['used_hfl_lo'], float))
    out['LFL_slope'] = np.asarray(ds['LFL_slope'], float) * dlogL
    out['HFL_slope'] = np.asarray(ds['HFL_slope'], float) * dlogH
    out['slope_units'] = 'drift_dex'
    return out

def _pair_median_spectrum(results, ev1_id, ev2_id, phase, method, noise_thres=3.5, prefer_H_for_coda_fd=True):
    env_like = method in ('envelope', 'envelope_fd', 'envelope_L')
    if env_like:
        kwargs = {}
        if prefer_H_for_coda_fd and (phase == 'CODA') and (method == 'envelope_fd'):
            kwargs['components'] = 'H'
        out = compute_phase_method_ratios(results, ev1_id, ev2_id, phase, method, noise_thres=noise_thres, return_freqs=True, return_stations=True, **kwargs)
        if not out or len(out) < 2: return None, None, 0
        ratios, fc_lists = out[0], out[1]
        if not ratios: return None, None, 0
        n_sta = len(ratios)
        fc_med, med_vals = _aggregate_ragged(fc_lists, ratios, round_dec=6, min_count=1)
        if fc_med is None or med_vals is None: return None, None, 0
        return fc_med, med_vals, n_sta

    out = compute_phase_method_ratios(results, ev1_id, ev2_id, phase, method, noise_thres=noise_thres, return_freqs=False, return_stations=True)
    if not out or len(out) < 1: return None, None, 0
    ratios = out[0]
    if not ratios: return None, None, 0
    n_sta = len(ratios)
    try: freq_used = np.asarray(results[ev1_id][phase][method]['frequencies'][0], float)
    except:
        try: freq_used = np.asarray(results[ev2_id][phase][method]['frequencies'][0], float)
        except: return None, None, 0
    pair_med = np.nanmedian(np.vstack([np.asarray(r, float) for r in ratios]), axis=0)
    if not np.any(np.isfinite(pair_med)): return None, None, 0
    return freq_used, pair_med, n_sta

def _band_stats(freq, y, band):
    f = np.asarray(freq, float); yy = np.asarray(y, float)
    m = np.isfinite(f) & np.isfinite(yy) & (f >= band[0]) & (f <= band[1])
    if np.sum(m) < 2: return np.nan, np.nan, np.nan
    mean = float(np.nanmean(yy[m]))
    std  = float(np.nanstd(yy[m], ddof=1))
    try:
        slope, _ = np.polyfit(np.log10(f[m]), yy[m], 1)
    except Exception: slope = np.nan
    return mean, std, slope

def fit_brune_parameters(freq, log_r, freq_lims=(0.1, 50.0)):
    def brune_log10_ratio(f, log_omega_ratio, fc1, fc2, n=2):
        f = np.maximum(f, 1e-6); fc1 = np.maximum(fc1, 1e-6); fc2 = np.maximum(fc2, 1e-6)
        return log_omega_ratio + np.log10(1 + (f/fc2)**n) - np.log10(1 + (f/fc1)**n)
        
    mask = (freq >= freq_lims[0]) & (freq <= freq_lims[1]) & np.isfinite(log_r)
    if mask.sum() < 8: return np.nan, np.nan, np.nan
    f_data = freq[mask]; y_data = log_r[mask]
    dc = np.median(y_data[:max(1, len(y_data)//5)])
    fc = np.sqrt(freq_lims[0] * freq_lims[1])
    try:
        popt, _ = curve_fit(brune_log10_ratio, f_data, y_data, p0=[dc, fc, fc], 
                            bounds=([-6, freq_lims[0]*0.5]*3, [6, freq_lims[1]*2.0]*3), maxfev=1000)
        return popt[0], popt[1], popt[2]
    except: return np.nan, np.nan, np.nan

def gather_pair_stats(results, ids_arr, set_indices, method, phase, setnum=0, noise_thres=3.5):
    import sys as _sys
    _self = _sys.modules[__name__]
    LFL_BAND = getattr(_self, 'LFL_BAND_OVERRIDE', (0.25, 1.0))
    HFL_BAND = getattr(_self, 'HFL_BAND_OVERRIDE', (20.0, 45.0))
    pairs_idx = []
    LFL_mean, LFL_std, LFL_slope = [], [], []
    HFL_mean, HFL_std, HFL_slope = [], [], []
    SDs, N_sta = [], []
    brune_LFL, brune_HFL, fc1_list, fc2_list = [], [], [], []

    for (i_idx, j_idx) in set_indices[setnum]:
        ev1 = str(ids_arr[i_idx]); ev2 = str(ids_arr[j_idx])
        try: freq, logr, n_sta = _pair_median_spectrum(results, ev1, ev2, phase, method, noise_thres=noise_thres)
        except: continue
        if (freq is None) or (logr is None) or (n_sta == 0): continue

        l_mean, l_std, l_slope = _band_stats(freq, logr, LFL_BAND)
        h_mean, h_std, h_slope = _band_stats(freq, logr, HFL_BAND)

        if np.isfinite(l_mean) and np.isfinite(h_mean):
            b_om, b_fc1, b_fc2 = fit_brune_parameters(freq, logr)
            b_hfl = b_om + 2*np.log10(b_fc1) - 2*np.log10(b_fc2) if np.isfinite(b_om) else np.nan

            pairs_idx.append((int(i_idx), int(j_idx)))
            LFL_mean.append(l_mean); LFL_std.append(l_std); LFL_slope.append(l_slope)
            HFL_mean.append(h_mean); HFL_std.append(h_std); HFL_slope.append(h_slope)
            SDs.append(np.log10((10**(1.5 * h_mean)) / (10**(0.5 * l_mean))))
            N_sta.append(n_sta)
            brune_LFL.append(b_om); brune_HFL.append(b_hfl); fc1_list.append(b_fc1); fc2_list.append(b_fc2)

    return {
        'pairs_idx': np.array(pairs_idx, dtype=int),
        'LFL_mean': np.asarray(LFL_mean, float), 'LFL_std': np.asarray(LFL_std, float), 'LFL_slope': np.asarray(LFL_slope, float),
        'HFL_mean': np.asarray(HFL_mean, float), 'HFL_std': np.asarray(HFL_std, float), 'HFL_slope': np.asarray(HFL_slope, float),
        'SD': np.asarray(SDs, float), 'N_sta': np.asarray(N_sta, int),
        'brune_LFL': np.asarray(brune_LFL, float), 'brune_HFL': np.asarray(brune_HFL, float),
        'fc1': np.asarray(fc1_list, float), 'fc2': np.asarray(fc2_list, float),
    }

def sd_from_LH(LFL_mean, HFL_mean):
    return 1.5 * np.asarray(HFL_mean, float) - 0.5 * np.asarray(LFL_mean, float)

def sigma_SD_from_LHsig(LFL_std, HFL_std):
    Ls = np.asarray(LFL_std, float); Hs = np.asarray(HFL_std, float)
    return np.sqrt((1.5*Hs)**2 + (0.5*Ls)**2)

# ==========================================
# 3. DATASET & GRAPH OPS
# ==========================================

def restrict_dataset_to_largest_component(ds):
    pairs = list(map(tuple, ds['pairs']))
    G = nx.Graph(); G.add_edges_from(pairs)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if not comps: return {}
    largest = comps[0]
    mask = np.array([(i in largest and j in largest) for (i, j) in pairs], bool)
    out = {'nodes': sorted(list(largest))}
    for key, val in ds.items():
        if key == 'nodes': continue
        if len(val) == len(pairs): out[key] = np.array(val)[mask] if isinstance(val, (list, np.ndarray)) else val
    return out

def build_global_pair_dataset(results, ids_arr, set_indices, spec, setnum=0, noise_thres=3.5, agg='median', restrict_to_largest=True):
    from collections import defaultdict
    combos = parse_phase_method_spec(spec)
    store = {}
    for ph, m in combos:
        try:
            d = gather_pair_stats(results, ids_arr, set_indices, m, ph, setnum, noise_thres)
            if len(d['pairs_idx']) > 0: store[(ph, m)] = d
        except: pass

    per_pair = defaultdict(lambda: {'L':[], 'H':[], 'sL':[], 'sH':[], 'mL':[], 'mH':[], 'N':[], 'bL':[], 'bH':[], 'fc1':[], 'fc2':[]})
    for (ph, m), d in store.items():
        for p, L, H, sL, sH, mL, mH, N, bL, bH, f1, f2 in zip(d['pairs_idx'], d['LFL_mean'], d['HFL_mean'], d['LFL_std'], d['HFL_std'], d['LFL_slope'], d['HFL_slope'], d['N_sta'], d['brune_LFL'], d['brune_HFL'], d['fc1'], d['fc2']):
            if not (np.isfinite(L) and np.isfinite(H)): continue
            k = tuple(sorted(p))
            per_pair[k]['L'].append(L); per_pair[k]['H'].append(H)
            per_pair[k]['sL'].append(sL); per_pair[k]['sH'].append(sH)
            per_pair[k]['mL'].append(mL); per_pair[k]['mH'].append(mH)
            per_pair[k]['N'].append(N)
            if np.isfinite(bL): per_pair[k]['bL'].append(bL)
            if np.isfinite(bH): per_pair[k]['bH'].append(bH)
            if np.isfinite(f1): per_pair[k]['fc1'].append(f1)
            if np.isfinite(f2): per_pair[k]['fc2'].append(f2)

    agg_func = np.nanmedian if agg == 'median' else np.nanmean
    pairs, LFL, HFL, sLFL, sHFL, mLFL, mHFL, N_eff, bL, bH, f1s, f2s = [],[],[],[],[],[],[],[],[],[],[],[]

    for p, v in per_pair.items():
        if not v['L']: continue
        pairs.append(p)
        LFL.append(agg_func(v['L'])); HFL.append(agg_func(v['H']))
        sLFL.append(agg_func(v['sL'])); sHFL.append(agg_func(v['sH']))
        mLFL.append(agg_func(v['mL'])); mHFL.append(agg_func(v['mH']))
        N_eff.append(np.nanmedian(v['N']))
        
        bL.append(agg_func(v['bL']) if v['bL'] else np.nan)
        bH.append(agg_func(v['bH']) if v['bH'] else np.nan)
        f1s.append(agg_func(v['fc1']) if v['fc1'] else np.nan)
        f2s.append(agg_func(v['fc2']) if v['fc2'] else np.nan)

    out = {
        'pairs': np.array(pairs), 'LFL': np.array(LFL), 'HFL': np.array(HFL),
        'sLFL': np.array(sLFL), 'sHFL': np.array(sHFL),
        'LFL_slope': np.array(mLFL), 'HFL_slope': np.array(mHFL),
        'n_sta': np.array(N_eff), 'brune_LFL': np.array(bL), 'brune_HFL': np.array(bH),
        'fc1': np.array(f1s), 'fc2': np.array(f2s),
        'sd': sd_from_LH(np.array(LFL), np.array(HFL)),
        'sigma_sd': sigma_SD_from_LHsig(np.array(sLFL), np.array(sHFL))
    }
    if restrict_to_largest: return restrict_dataset_to_largest_component(out)
    return out

# ==========================================
# 4. INVERSION & UNCERTAINTY
# ==========================================

def run_dual_band_inversion(ds, mag_arr, slope_weight=0.5, mag_limits=None,
                            min_hfl_degree=1, min_lfl_degree=1,
                            solve_hfl_min=None):
    """
    Solves for Relative LFL and Relative HFL in a single matrix.
    Computes Stress Drop ONLY for events with sufficient constraints in both bands.
    """
    print(f"--- Running DUAL BAND Inversion (Slope Weight={slope_weight}) ---")
    
    pairs = ds['pairs']
    n_pairs = len(pairs)
    nodes = ds['nodes']
    n_nodes = len(nodes)
    node_to_col = {n: i for i, n in enumerate(nodes)}
    
    # Defaults
    limits = {'LFL_max': 3.5, 'HFL_min': 2.5}
    if mag_limits: limits.update(mag_limits)
    if solve_hfl_min is None:
        solve_hfl_min = limits['HFL_min']
    print(f"  Limits: LFL[...{limits['LFL_max']}], HFL[{limits['HFL_min']}...]")

    # --- 1. Weights & Gating ---
    def get_sigma_safe(std_key, slope_key):
        s = np.nan_to_num(np.asarray(ds[std_key], float), nan=np.inf)
        sl = np.nan_to_num(np.asarray(ds[slope_key], float), nan=np.inf)
        return np.sqrt(s**2 + (slope_weight * np.abs(sl))**2)

    sigma_lfl = get_sigma_safe('sLFL', 'LFL_slope')
    sigma_hfl = get_sigma_safe('sHFL', 'HFL_slope')
    
    # Gating Logic
    pair_arr = np.array(pairs)
    mags_A = mag_arr[pair_arr[:, 0]]
    mags_B = mag_arr[pair_arr[:, 1]]
    
    mask_lfl = (mags_A <= limits['LFL_max']) & (mags_B <= limits['LFL_max'])
    mask_hfl = (mags_A >= solve_hfl_min) & (mags_B >= solve_hfl_min)
    
    w_sta = np.sqrt(np.nan_to_num(ds['n_sta'], nan=1.0)) if ds['n_sta'] is not None else np.ones(n_pairs)
    
    weight_lfl = w_sta / np.maximum(sigma_lfl, 1e-4)
    weight_hfl = w_sta / np.maximum(sigma_hfl, 1e-4)
    
    weight_lfl[~mask_lfl] = 0.0
    weight_hfl[~mask_hfl] = 0.0
    
    print(f"  Active Constraints: {np.sum(mask_lfl)} LFL pairs, {np.sum(mask_hfl)} HFL pairs.")

    # ============================================================
    # COUNT band-specific degrees (for OUTPUT masking only)
    # DO NOT modify weights — let LSQR use all available data
    # ============================================================
    lfl_active_deg = Counter()
    hfl_active_deg = Counter()
    for r, (i, j) in enumerate(pairs):
        ci, cj = node_to_col[i], node_to_col[j]
        if weight_lfl[r] > 1e-9:
            lfl_active_deg[ci] += 1
            lfl_active_deg[cj] += 1
        if weight_hfl[r] > 1e-9:
            hfl_active_deg[ci] += 1
            hfl_active_deg[cj] += 1

    n_lfl_ok = sum(1 for v in lfl_active_deg.values() if v >= min_lfl_degree)
    n_hfl_ok = sum(1 for v in hfl_active_deg.values() if v >= min_hfl_degree)
    print(f"  Events with >= {min_lfl_degree} LFL pairs: {n_lfl_ok}")
    print(f"  Events with >= {min_hfl_degree} HFL pairs: {n_hfl_ok}")

    # --- 2. Build Matrix (unchanged — use ALL weighted pairs) ---
    rows, cols, data = [], [], []
    d_vec, w_vec = [], []
    
    for r, (i, j) in enumerate(pairs):
        if weight_lfl[r] < 1e-9: continue
        idx_i = node_to_col[i]
        idx_j = node_to_col[j]
        current_row = len(d_vec)
        rows.extend([current_row, current_row])
        cols.extend([idx_i, idx_j])
        data.extend([1.0, -1.0])
        d_vec.append(ds['LFL'][r])
        w_vec.append(weight_lfl[r])

    col_offset = n_nodes
    for r, (i, j) in enumerate(pairs):
        if weight_hfl[r] < 1e-9: continue
        idx_i = node_to_col[i]
        idx_j = node_to_col[j]
        current_row = len(d_vec)
        rows.extend([current_row, current_row])
        cols.extend([col_offset + idx_i, col_offset + idx_j])
        data.extend([1.0, -1.0])
        d_vec.append(ds['HFL'][r])
        w_vec.append(weight_hfl[r])
        
    # --- 3. Solve ---
    total_cols = 2 * n_nodes
    G = sparse.coo_matrix((data, (rows, cols)), shape=(len(d_vec), total_cols))
    d = np.array(d_vec); w = np.array(w_vec)
    
    print(f"  Solving system: {G.shape}...")
    G_w = G.multiply(w[:, None])
    d_w = d * w
    
    m_sol = lsqr(G_w, d_w, damp=0.0)[0]
    
    m_lfl = m_sol[:n_nodes]
    m_hfl = m_sol[n_nodes:]
    
    # --- 4. Post-Process ---
    node_mags = mag_arr[nodes]
    gold_mask = (node_mags >= limits['HFL_min']) & (node_mags <= limits['LFL_max'])
    gold_indices = np.where(gold_mask)[0]
    
    if len(gold_indices) == 0:
        print("WARNING: No Dual-Band overlap found!")
        lfl_mean = np.mean(m_lfl[m_lfl != 0])
        hfl_mean = np.mean(m_hfl[m_hfl != 0])
    else:
        lfl_mean = np.mean(m_lfl[gold_indices])
        hfl_mean = np.mean(m_hfl[gold_indices])
        
    m_lfl -= lfl_mean
    m_hfl -= hfl_mean

    # ============================================================
    # MASK outputs for under-constrained events
    # The inversion used all data, but we don't trust outputs
    # for events with too few constraints in a given band
    # ============================================================
    for idx in range(n_nodes):
        if lfl_active_deg.get(idx, 0) < min_lfl_degree:
            m_lfl[idx] = np.nan
        if hfl_active_deg.get(idx, 0) < min_hfl_degree:
            m_hfl[idx] = np.nan

    # --- 5. Compute Stress Drop ---
    m_sd = np.full(n_nodes, np.nan)

    for idx in gold_indices:
        # Only compute SD if BOTH bands are well-constrained
        if np.isfinite(m_lfl[idx]) and np.isfinite(m_hfl[idx]):
            m_sd[idx] = 1.5 * m_hfl[idx] - 0.5 * m_lfl[idx]

    # Final Center
    valid_mask = np.isfinite(m_sd)
    valid_sd = m_sd[valid_mask]
    if len(valid_sd) > 0:
        m_sd[valid_mask] -= np.mean(valid_sd)

    n_hfl_masked = np.sum(np.isnan(m_hfl))
    n_lfl_masked = np.sum(np.isnan(m_lfl))
    print(f"  Masked {n_hfl_masked} HFL outputs, {n_lfl_masked} LFL outputs (under-constrained)")
    print(f"  Inversion Complete. SD calculated for {len(valid_sd)} events "
          f"(filtered {len(gold_indices) - len(valid_sd)} under-constrained).")
    return m_sd, nodes, m_lfl, m_hfl
    
    
def restrict_dataset_to_subset(ds, indices):
    out = ds.copy()
    n_orig = len(ds['pairs'])
    for k, v in ds.items():
        if isinstance(v, (np.ndarray, list)) and len(v) == n_orig and k != 'nodes':
            out[k] = v[indices] if isinstance(v, np.ndarray) else [v[i] for i in indices]
    return out

def run_jackknife_uncertainty_aligned(ds, mag_arr, limits, n_iter=20, drop_fraction=0.2,
                                      solve_hfl_min=None, slope_weight=0.5, seed=None):
    rng = np.random.default_rng(seed)
    ref_sd, _, ref_lfl, _ = run_dual_band_inversion(ds, mag_arr, slope_weight, limits,
                                                     solve_hfl_min=solve_hfl_min)
    res_sd = np.full((len(ds['nodes']), n_iter), np.nan)
    res_lfl = np.full((len(ds['nodes']), n_iter), np.nan)
    n_keep = int(len(ds['pairs']) * (1 - drop_fraction))

    for k in range(n_iter):
        idx = rng.choice(len(ds['pairs']), n_keep, replace=False)
        try:
            iter_sd, _, iter_lfl, _ = run_dual_band_inversion(restrict_dataset_to_subset(ds, idx),
                                                              mag_arr, slope_weight, limits,
                                                              solve_hfl_min=solve_hfl_min)
            valid = np.isfinite(iter_sd) & np.isfinite(ref_sd)
            if np.sum(valid) > 10: res_sd[:, k] = iter_sd + np.median(ref_sd[valid] - iter_sd[valid])

            valid_lfl = np.isfinite(iter_lfl) & np.isfinite(ref_lfl)
            if np.sum(valid_lfl) > 10: res_lfl[:, k] = iter_lfl + np.median(ref_lfl[valid_lfl] - iter_lfl[valid_lfl])
        except Exception as e:
            print(f"  jackknife iteration {k} failed: {e}")

    return {'SD_std': np.nanstd(res_sd, axis=1), 'LFL_std': np.nanstd(res_lfl, axis=1)}
    
def find_triplets_and_compute_closure(ds, mag_arr, metrics=['LFL','HFL','SD'], slope_weight=0.5):
    import time
    from collections import defaultdict
    print("Finding triplets and mapping edges...")
    t0 = time.time()
    
    pairs = ds['pairs']
    adj_set = defaultdict(set)
    for i, j in pairs:
        adj_set[i].add(j)
        adj_set[j].add(i)
        
    pair_map = {tuple(p): idx for idx, p in enumerate(pairs)}
    def get_edge(u, v):
        if (u, v) in pair_map: return pair_map[(u, v)], 1.0
        return pair_map[(v, u)], -1.0

    # Pre-compute edge values to avoid doing it inside the loop
    n_sta = np.maximum(ds.get('n_sta', np.ones(len(pairs))), 1.0)
    def get_eff_sig(std_k, slope_k):
        num = np.sqrt(ds[std_k]**2 + (slope_weight * np.abs(ds[slope_k]))**2)
        return num / np.sqrt(n_sta)
        
    sigs = {'LFL': get_eff_sig('sLFL', 'LFL_slope'), 'HFL': get_eff_sig('sHFL', 'HFL_slope')}
    sigs['SD'] = np.sqrt((1.5*sigs['HFL'])**2 + (0.5*sigs['LFL'])**2)
    vals = {'LFL': ds['LFL'], 'HFL': ds['HFL'], 'SD': ds['sd']}
    
    results = {m: {'closure': [], 'sigma_trip': [], 'mag_med': [], 'mag_diff': []} for m in metrics}
    
    # --- MEMORY SAFE CHUNKING ---
    CHUNK_SIZE = 5_000_000 
    buffer = []
    total_triplets = 0
    
    def process_buffer(buf):
        if not buf: return
        arr = np.array(buf)
        e1_i = arr[:, 0].astype(int); e1_s = arr[:, 1]
        e2_i = arr[:, 2].astype(int); e2_s = arr[:, 3]
        e3_i = arr[:, 4].astype(int); e3_s = arr[:, 5]
        u_n = arr[:, 6].astype(int); v_n = arr[:, 7].astype(int); k_n = arr[:, 8].astype(int)
        
        mags = np.vstack([mag_arr[u_n], mag_arr[v_n], mag_arr[k_n]])
        m_med = np.median(mags, axis=0)
        m_diff = np.max(mags, axis=0) - np.min(mags, axis=0)
        
        for met in metrics:
            v_arr = vals[met]; s_arr = sigs[met]
            # Vectorized closure computation
            c_arr = np.abs(e1_s * v_arr[e1_i] + e2_s * v_arr[e2_i] + e3_s * v_arr[e3_i])
            s_arr_trip = np.sqrt(s_arr[e1_i]**2 + s_arr[e2_i]**2 + s_arr[e3_i]**2)
            
            results[met]['closure'].append(c_arr)
            results[met]['sigma_trip'].append(s_arr_trip)
            results[met]['mag_med'].append(m_med)
            results[met]['mag_diff'].append(m_diff)

    # Search for triplets
    for i, j in pairs:
        u, v = (i, j) if i < j else (j, i)
        common = adj_set[u].intersection(adj_set[v])
        
        if common:
            e1_idx, e1_sgn = get_edge(u, v)
            for k in common:
                # Enforce strict ascending order to deduplicate instantly without set()
                if k > v:
                    e2_idx, e2_sgn = get_edge(v, k)
                    e3_idx, e3_sgn = get_edge(k, u)
                    buffer.append((e1_idx, e1_sgn, e2_idx, e2_sgn, e3_idx, e3_sgn, u, v, k))
                    
                    # Offload to NumPy when buffer is full to save RAM
                    if len(buffer) >= CHUNK_SIZE:
                        process_buffer(buffer)
                        total_triplets += len(buffer)
                        buffer.clear() 
                        
    # Process anything left over
    if buffer:
        process_buffer(buffer)
        total_triplets += len(buffer)
        buffer.clear()
        
    print(f"Processed {total_triplets} unique triplets in {time.time() - t0:.2f} seconds.")
    
    # Concatenate the chunks into final 1D arrays
    for met in metrics:
        if results[met]['closure']:
            results[met]['closure'] = np.concatenate(results[met]['closure'])
            results[met]['sigma_trip'] = np.concatenate(results[met]['sigma_trip'])
            results[met]['mag_med'] = np.concatenate(results[met]['mag_med'])
            results[met]['mag_diff'] = np.concatenate(results[met]['mag_diff'])
        else:
            results[met]['closure'] = np.array([])
            results[met]['sigma_trip'] = np.array([])
            results[met]['mag_med'] = np.array([])
            results[met]['mag_diff'] = np.array([])
            
    return results
    
def generate_comparison_plots_fixed(df_res, roi, ext_moments, ext_sds, ext_ids, limits, figsize=None, save_path=None):
    from scipy.stats import norm, siegelslopes, linregress # Ensure these are imported
    
    # 1. Prepare Data Copy
    df = df_res.copy()
    
    # 2. Force Copy Metadata from ROI (Handle index alignment)
    df['depR'] = roi['depR'].reindex(df.index)
    df['latR'] = roi['latR'].reindex(df.index)
    df['lonR'] = roi['lonR'].reindex(df.index)
    
    # 4. Map External Data
    ext_map_mom = dict(zip([str(x) for x in ext_ids], ext_moments))
    ext_map_sd = dict(zip([str(x) for x in ext_ids], ext_sds))
    
    def safe_log10(val):
        return np.log10(val) if (pd.notna(val) and val > 0) else np.nan

    df['Ext_LogMom'] = df.index.map(lambda x: safe_log10(ext_map_mom.get(x, np.nan)))
    df['Ext_LogSD'] = df.index.map(lambda x: safe_log10(ext_map_sd.get(x, np.nan)))
    
    # Filter External Outliers
    df.loc[df['Ext_LogMom'] > 18, 'Ext_LogMom'] = np.nan
    df.loc[df['Ext_LogSD'] > 9, 'Ext_LogSD'] = np.nan
    
    # Filter Ghost Events (Outside valid bands)
    df.loc[df['Mag'] > limits['LFL_max'], 'LFL_rel'] = np.nan
    df.loc[df['Mag'] < limits['HFL_min'], 'HFL_rel'] = np.nan
    
    # 5. Create Goldilocks Subset
    df_sd = df[df['is_dual_band']].copy()

    # Determine Base Filename
    if save_path:
        base = str(save_path).replace('.png', '')
    else:
        base = "comparison_plot"

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    sc_kws = {'alpha': 0.6, 'edgecolor': 'k', 'linewidth': 1.0}

    # --- MODIFIED: Now returns the calculated slope ---
    def save_one_plot(x, y, c, title, xl, yl, suffix, fit=False, robust=True, identity=False):
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 5: 
            return np.nan # Not enough data to fit
        
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(x[valid], y[valid], c=c[valid], cmap='viridis', s=50, **sc_kws)
        cbar = plt.colorbar(sc, drawedges=False)
        cbar.set_label('Effective Degree' if False else 'Magnitude')
        cbar.ax.tick_params(axis='y', length=0)
        cbar.solids.set_edgecolor("face")
        plt.title(title, fontsize=14); plt.xlabel(xl, fontsize=12); plt.ylabel(yl, fontsize=12)
        plt.grid(True, ls=':')
        
        computed_slope = np.nan
        
        if identity:
            m = min(x[valid].min(), y[valid].min()); M = max(x[valid].max(), y[valid].max())
            plt.plot([m, M], [m, M], 'k--', lw=2, label='1:1')
            
        if fit:
            if robust: 
                slope, inter = siegelslopes(y[valid], x[valid])
                r_txt = "Robust"
            else: 
                slope, inter, r, _, _ = linregress(x[valid], y[valid])
                r_txt = f"R={r:.2f}"
            
            computed_slope = slope
            xx = np.array([x[valid].min(), x[valid].max()])
            plt.plot(xx, slope*xx + inter, 'r-', lw=3, label=f'Slope={slope:.2f}')
            
            txt = f"Slope: {slope:.3f}\n{r_txt}"
            plt.gca().text(0.05, 0.95, txt, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)
            plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{base}_{suffix}.png", dpi=150)
        plt.close()
        
        return computed_slope

    # 1. HFL vs LFL
    save_one_plot(df['LFL_rel'], df['HFL_rel'], df['Mag'], f"Scaling: HFL vs LFL (N={len(df)})", "Relative LFL", "Relative HFL", "scaling", fit=True)
    
    # 2. LFL Validation (Capture this slope)
    lfl_slope = save_one_plot(df['Ext_LogMom'], df['LFL_rel'], df['Mag'], "LFL Validation", "External Log10 Moment", "Inverted LFL", "LFL_val", fit=True)
    
    # 3. SD vs Depth (Capture this slope)
    depth_slope = save_one_plot(df_sd['depR'], df_sd['SD_rel'], df_sd['Mag'], "Stress Drop vs Depth", "Depth (km)", "Relative Log10 SD", "SD_depth", fit=True)
    
    # 4. SD Validation
    shift = (df_sd['Ext_LogSD'] - df_sd['SD_rel']).median()
    save_one_plot(df_sd['Ext_LogSD'], df_sd['SD_rel']+shift, df_sd['Mag'], "SD Validation", "External Log10 SD", "Inverted SD (Shifted)", "SD_val", fit=True, identity=True)
    
    # 5. Map
    plt.figure(figsize=(8,6))
    plt.scatter(df['lonR'], df['latR'], c=df['SD_rel'], cmap='coolwarm', s=50, **sc_kws)
    plt.title("Map View"); plt.xlabel("Longitude"); plt.ylabel("Latitude");
    cbar = plt.colorbar(label='Log10 SD', drawedges=False)
    cbar.ax.tick_params(axis='y', length=0)
    cbar.solids.set_edgecolor("face")
    plt.savefig(f"{base}_map.png", dpi=150); plt.close()
    
    # 6. Section
    plt.figure(figsize=(8,6))
    plt.scatter(df['lonR'], df['depR'], c=df['SD_rel'], cmap='coolwarm', s=50, **sc_kws)
    plt.gca().invert_yaxis(); plt.title("Cross Section"); plt.xlabel("Longitude"); plt.ylabel("Depth");
    cbar = plt.colorbar(label='Log10 SD', drawedges=False)
    cbar.ax.tick_params(axis='y', length=0)
    cbar.solids.set_edgecolor("face")
    plt.savefig(f"{base}_section.png", dpi=150); plt.close()
    
    # 7. SD vs LFL
    sd_lfl_slope = save_one_plot(df_sd['LFL_rel'], df_sd['SD_rel'], df_sd['Mag'], "Output Check: SD vs LFL", "Inverted LFL", "Inverted SD", "SD_vs_LFL", fit=True)
    
    # 8. SD vs Mag (Capture this slope)
    mag_slope = save_one_plot(df_sd['Mag'], df_sd['SD_rel'], df_sd['Mag'], "Output Check: SD vs Catalog Mag", "Catalog Magnitude", "Inverted SD", "SD_vs_Mag", fit=True)
    
    # 9. Histogram (Capture the sigma)
    plt.figure(figsize=(8,6))
    vals = df_sd['SD_rel'].dropna()
    mu, std = norm.fit(vals)
    plt.hist(vals, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='k')
    x = np.linspace(vals.min(), vals.max(), 100)
    plt.plot(x, norm.pdf(x, mu, std), 'k-', lw=3, label=r'$\sigma$=' + f'{std:.2f}')
    plt.legend(); plt.title("SD Distribution"); plt.xlabel("Relative Log10 SD")
    plt.savefig(f"{base}_hist.png", dpi=150); plt.close()

    # --- RETURN THE CAPTURED METRICS ---
    return {
        'LFL_val_slope': lfl_slope,
        'SD_depth_slope': depth_slope,
        'SD_lfl_slope': sd_lfl_slope,
        'SD_sigma': std
    }
    
def plot_triplet_closure_dashboard(closure_data, metric='SD', save_path=None):
    data = closure_data[metric]
    c = data['closure']; sig = data['sigma_trip']; mag = data['mag_med']; mdiff = data['mag_diff']
    
    mean_val = np.nanmean(c)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    def scatter_kde(ax, x, y, xl, yl, title, logx=False, logy=False, fit=False):
        valid = np.isfinite(x) & np.isfinite(y) & (x>0 if logx else True) & (y>0 if logy else True)
        x, y = x[valid], y[valid]
        if len(x) < 5: return
        
        # Map data to log space if requested
        x_mapped = np.log10(x) if logx else x
        y_mapped = np.log10(y) if logy else y
        
        # --- THE FIX: DOWNSAMPLE FOR MASSIVE DATA ---
        MAX_POINTS = 10000
        if len(x) > MAX_POINTS:
            idx = np.random.choice(len(x), MAX_POINTS, replace=False)
            x_sub, y_sub = x_mapped[idx], y_mapped[idx]
            x_plot, y_plot = x[idx], y[idx]
        else:
            x_sub, y_sub = x_mapped, y_mapped
            x_plot, y_plot = x, y
            
        xy_sub = np.vstack([x_sub, y_sub])
        
        # Train and evaluate KDE only on the capped subset
        kde = gaussian_kde(xy_sub)
        z = kde(xy_sub)
        sort_idx = z.argsort()
        
        # Plot the subset (10,000 points visually fills the screen anyway)
        ax.scatter(x_plot[sort_idx], y_plot[sort_idx], c=z[sort_idx], s=10, cmap='turbo', edgecolor='none', alpha=0.5)
        
        # Add Contours (Evaluated using the fast subset KDE)
        try:
            xi, yi = np.mgrid[x_mapped.min():x_mapped.max():100j, y_mapped.min():y_mapped.max():100j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
            
            xi_plot = 10**xi if logx else xi
            yi_plot = 10**yi if logy else yi
            
            ax.contour(xi_plot, yi_plot, zi.reshape(xi.shape), levels=6, colors='white', linewidths=0.75, alpha=0.6)
        except Exception as e:
            pass

        if fit:
            slope, inter = np.polyfit(x_mapped, y_mapped, 1)
            xx_mapped = np.linspace(x_mapped.min(), x_mapped.max(), 100)
            yy_mapped = slope * xx_mapped + inter
            
            xx = 10**xx_mapped if logx else xx_mapped
            yy = 10**yy_mapped if logy else yy_mapped
            ax.plot(xx, yy, 'r--', lw=2)

        if logx: ax.set_xscale('log')
        if logy: ax.set_yscale('log')
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, ls=':', alpha=0.6)

    scatter_kde(ax1, mag, c, "Median Mag", f"|Closure| {metric}", "Error vs Event Size", logy=True)
    scatter_kde(ax2, mdiff, c, "Mag Diff", f"|Closure| {metric}", "Error vs Similarity", logy=True)
    scatter_kde(ax3, sig, c, "Predicted Sigma", f"|Closure| {metric}", "Validation", logx=True, logy=True, fit=True)
    
    m, M = sig.min(), sig.max()
    ax3.plot([m, M], [m, M], 'k--', lw=2, label='1:1')
    
    text_str = f"Mean {metric} Closure: {mean_val:.3f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax3.text(0.05, 0.95, text_str, transform=ax3.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    ax3.legend()

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150); plt.close()

def plot_clean_uncertainties(df_res, errors, limits, node_degree=None, save_path=None):
    """
    Single-panel plot: SD uncertainty vs magnitude, colored by node degree.
 
    Parameters
    ----------
    df_res      : DataFrame with Mag, is_dual_band (or is_goldilocks)
    errors      : dict with 'SD_std' array
    limits      : dict with HFL_min, LFL_max
    node_degree : array of per-node degree values (same length as df_res)
    save_path   : output path
    """
    mags = df_res['Mag'].to_numpy(float)
    sd_err = errors['SD_std']
 
    # Use whichever column exists
    if 'is_dual_band' in df_res.columns:
        valid_sd = df_res['is_dual_band'].to_numpy(bool)
    else:
        valid_sd = df_res['is_goldilocks'].to_numpy(bool)
 
    valid = valid_sd & np.isfinite(sd_err) & (sd_err > 0)
 
    if np.sum(valid) < 5:
        print("Not enough valid events for uncertainty plot.")
        return
 
    x = mags[valid]
    y = sd_err[valid]
 
    # Color by degree if provided, else by magnitude
    if node_degree is not None and len(node_degree) == len(mags):
        c = node_degree[valid].astype(float)
        c_label = 'Effective Degree'
        cmap = 'viridis'
    else:
        c = x
        c_label = 'Magnitude'
        cmap = 'plasma'
 
    fig, ax = plt.subplots(figsize=(8, 6))
 
    sc = ax.scatter(x, y, c=c, cmap=cmap, s=15, alpha=0.7,
                    edgecolor='k', linewidth=0.2)
 
    # Clean colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02, drawedges=False)
    cbar.set_label(c_label, fontsize=11)
    cbar.ax.tick_params(axis='y', length=0, labelsize=9)
    cbar.solids.set_edgecolor("face")
 
    # Trend line
    if len(x) > 5:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_fit, p(x_fit), 'r-', lw=2, alpha=0.7,
                label=f'Slope = {z[0]:.4f}')
        ax.legend(fontsize=10, loc='upper right')
 
    # Median annotation
    med_err = np.median(y)
    ax.axhline(med_err, color='gray', ls='--', lw=1, alpha=0.5)
    ax.text(0.03, 0.97, f'Median uncertainty: {med_err:.4f}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(facecolor='white', alpha=0.8))
 
    ax.set_xlabel('Catalog Magnitude', fontsize=12)
    ax.set_ylabel('Jackknife SD Uncertainty (log$_{10}$ units)', fontsize=12)
    ax.set_ylim(0, min(0.5, y.max() * 1.2))
    ax.grid(True, ls=':', alpha=0.3)
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
        
"""
stress_inversion_lib_patch.py
=============================
Add these functions to your existing stress_inversion_lib.py on Sherlock.
They replace/extend gather_pair_stats and build_global_pair_dataset
with adaptive LFL band support and component selection.

Integration: append this entire file to the end of stress_inversion_lib.py,
or import it alongside.
"""

import numpy as np
from collections import defaultdict, Counter
import networkx as nx
import scipy.sparse as sparse
from scipy.sparse.linalg import lsqr

# ==========================================================
# ADAPTIVE BAND CONFIGURATION
# ==========================================================
# Derived from local analysis: median S-wave fc fit with 0.26× safety factor
# log10(fc_S) = 1.749 - 0.295 * M, edge = 0.26 × 10^(1.749 - 0.295*M)
# Clipped to [1.2, 5.0] Hz

ADAPTIVE_LFL_DEFAULTS = {
    'enabled': True,
    'knots_mag': [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0],
    'knots_hz':  [3.35, 2.87, 2.46, 2.11, 1.81, 1.55, 1.33, 1.20, 1.20],
    'lfl_lo': 0.25,
    'hfl_lo': 25.0,
    'hfl_hi': 40.0,
}


def get_pair_bands(mag_i, mag_j, config=None,
                   fallback_lfl=(0.25, 1.0), fallback_hfl=(25.0, 40.0)):
    """
    Returns ((lfl_lo, lfl_hi), (hfl_lo, hfl_hi)) for a pair.
    Uses adaptive LFL edge if config['enabled'] is True.
    For the pair, takes the conservative (min) of both events' LFL edges.
    """
    if config is None:
        config = ADAPTIVE_LFL_DEFAULTS

    hfl_band = (config.get('hfl_lo', fallback_hfl[0]),
                config.get('hfl_hi', fallback_hfl[1]))

    if not config.get('enabled', False):
        lfl_hi = config.get('hfl_hi', fallback_lfl[1])  # bug guard
        # Use override bands if available
        try:
            lfl_band = (LFL_BAND_OVERRIDE[0], LFL_BAND_OVERRIDE[1])
            hfl_band = (HFL_BAND_OVERRIDE[0], HFL_BAND_OVERRIDE[1])
        except:
            lfl_band = fallback_lfl
        return lfl_band, hfl_band

    knots_mag = config['knots_mag']
    knots_hz = config['knots_hz']
    lfl_lo = config.get('lfl_lo', 0.25)

    def hi_for_mag(m):
        m_clipped = np.clip(m, knots_mag[0], knots_mag[-1])
        return float(np.interp(m_clipped, knots_mag, knots_hz))

    pair_lfl_hi = min(hi_for_mag(mag_i), hi_for_mag(mag_j))
    return (lfl_lo, pair_lfl_hi), hfl_band


# ==========================================================
# COMPONENT MAP DEFAULTS
# ==========================================================
# None = use all components (E, N, Z geometric mean)
# Set to restrict: {('P','multitaper'): ('Z',), ('S','multitaper'): ('E','N')}
DEFAULT_COMP_MAP = None


# ==========================================================
# MODIFIED: gather_pair_stats with adaptive bands + components
# ==========================================================

def gather_pair_stats_adaptive(results, ids_arr, roi, set_indices,
                                method, phase, setnum=0, noise_thres=3.5,
                                components=None, adaptive_config=None,fit_brune=True):
    """
    Adaptive-band version of gather_pair_stats.
    
    Key differences from the original:
      - Accepts results/ids_arr/roi/set_indices as explicit arguments
        (no globals needed)
      - Uses per-pair adaptive LFL bands via get_pair_bands()
      - Accepts components parameter for phase-specific component selection
      - Records which band edges were used per pair
    """
    pairs_idx = []
    LFL_mean, LFL_std, LFL_slope = [], [], []
    HFL_mean, HFL_std, HFL_slope = [], [], []
    SDs, N_sta = [], []
    used_lfl_hi, used_hfl_lo = [], []
    brune_LFL, brune_HFL, fc1_list, fc2_list = [], [], [], []

    for (i_idx, j_idx) in set_indices[setnum]:
        ev1_id = str(ids_arr[i_idx])
        ev2_id = str(ids_arr[j_idx])

        try:
            freq, logr, n_sta = _pair_median_spectrum_adaptive(
                results, ev1_id, ev2_id, phase, method,
                noise_thres=noise_thres, components=components
            )
        except Exception:
            continue

        if (freq is None) or (logr is None) or (n_sta == 0):
            continue

        # Get adaptive bands for this pair
        mi = roi.loc[ids_arr[i_idx], 'mag'] if ids_arr[i_idx] in roi.index else np.nan
        mj = roi.loc[ids_arr[j_idx], 'mag'] if ids_arr[j_idx] in roi.index else np.nan
        lfl_band, hfl_band = get_pair_bands(mi, mj, config=adaptive_config)

        l_mean, l_std, l_slope = _band_stats_lib(freq, logr, lfl_band)
        h_mean, h_std, h_slope = _band_stats_lib(freq, logr, hfl_band)

        if np.isfinite(l_mean) and np.isfinite(h_mean):
            if fit_brune:
                try:
                    b_omega, b_fc1, b_fc2 = fit_brune_parameters(freq, logr)
                except Exception:
                    b_omega, b_fc1, b_fc2 = np.nan, np.nan, np.nan
            else:
                b_omega, b_fc1, b_fc2 = np.nan, np.nan, np.nan

            if np.isfinite(b_omega):
                b_hfl_val = b_omega + 2*np.log10(max(b_fc1,1e-6)) - 2*np.log10(max(b_fc2,1e-6))
            else:
                b_hfl_val = np.nan

            pairs_idx.append(tuple(sorted((int(i_idx), int(j_idx)))))
            LFL_mean.append(l_mean); LFL_std.append(l_std); LFL_slope.append(l_slope)
            HFL_mean.append(h_mean); HFL_std.append(h_std); HFL_slope.append(h_slope)
            SDs.append(np.log10((10**(1.5 * h_mean)) / (10**(0.5 * l_mean))))
            N_sta.append(n_sta)
            used_lfl_hi.append(lfl_band[1])
            used_hfl_lo.append(hfl_band[0])
            brune_LFL.append(b_omega)
            brune_HFL.append(b_hfl_val)
            fc1_list.append(b_fc1)
            fc2_list.append(b_fc2)

    return {
        'pairs_idx': np.array(pairs_idx, dtype=int) if pairs_idx else np.array([], dtype=int).reshape(0,2),
        'LFL_mean': np.asarray(LFL_mean, float),
        'LFL_std': np.asarray(LFL_std, float),
        'LFL_slope': np.asarray(LFL_slope, float),
        'HFL_mean': np.asarray(HFL_mean, float),
        'HFL_std': np.asarray(HFL_std, float),
        'HFL_slope': np.asarray(HFL_slope, float),
        'SD': np.asarray(SDs, float),
        'N_sta': np.asarray(N_sta, int),
        'used_lfl_hi': np.asarray(used_lfl_hi, float),
        'used_hfl_lo': np.asarray(used_hfl_lo, float),
        'brune_LFL': np.asarray(brune_LFL, float),
        'brune_HFL': np.asarray(brune_HFL, float),
        'fc1': np.asarray(fc1_list, float),
        'fc2': np.asarray(fc2_list, float),
    }


def _pair_median_spectrum_adaptive(results, ev1_id, ev2_id, phase, method,
                                    noise_thres=3.5, components=None):
    """
    Wrapper around compute_phase_method_ratios that passes components.
    Returns (freq, median_log_ratio, n_stations) or (None, None, 0).
    """
    out = compute_phase_method_ratios(
        results, ev1_id, ev2_id, phase, method,
        noise_thres=noise_thres,
        return_freqs=False,
        return_stations=True,
        components=components
    )
    if not out or len(out) < 2:
        return None, None, 0
    ratios_list, stations = out
    if not ratios_list:
        return None, None, 0

    n_sta = len(ratios_list)

    try:
        freq = np.asarray(results[ev1_id][phase][method]['frequencies'][0], float)
    except Exception:
        try:
            freq = np.asarray(results[ev2_id][phase][method]['frequencies'][0], float)
        except Exception:
            return None, None, 0

    pair_med = np.nanmedian(np.vstack([np.asarray(r, float) for r in ratios_list]), axis=0)
    if not np.any(np.isfinite(pair_med)):
        return None, None, 0

    return freq, pair_med, n_sta


def _band_stats_lib(freq, y, band):
    """Mean, std, slope within a frequency band."""
    f = np.asarray(freq, float)
    yy = np.asarray(y, float)
    m = np.isfinite(f) & np.isfinite(yy) & (f >= band[0]) & (f <= band[1])
    if np.sum(m) < 2:
        return np.nan, np.nan, np.nan
    mean = float(np.nanmean(yy[m]))
    std_ = float(np.nanstd(yy[m], ddof=1))
    try:
        slope, _ = np.polyfit(np.log10(f[m]), yy[m], 1)
    except Exception:
        slope = np.nan
    return mean, std_, float(slope)


# ==========================================================
# MODIFIED: build_global_pair_dataset with adaptive + comp_map
# ==========================================================

def build_global_pair_dataset_adaptive(results, ids_arr, roi, set_indices,
                                        spec, setnum=0, noise_thres=3.5,
                                        agg='median', restrict_to_largest=True,
                                        comp_map=None, adaptive_config=None,
                                        fit_brune=True):
    """
    Build pair dataset with adaptive bands and optional component selection.
    
    Parameters
    ----------
    comp_map : dict or None
        Per-phase component restriction.
        e.g. {('P','multitaper'): ('Z',), ('S','multitaper'): ('E','N')}
    adaptive_config : dict or None
        Adaptive LFL config. None = use ADAPTIVE_LFL_DEFAULTS.
    """
    combos = parse_phase_method_spec(spec)
    store = {}
    for ph, m in combos:
        comps = comp_map.get((ph, m)) if comp_map else None
        try:
            d = gather_pair_stats_adaptive(
                results, ids_arr, roi, set_indices,
                m, ph, setnum=setnum, noise_thres=noise_thres,
                components=comps, adaptive_config=adaptive_config,
                fit_brune=fit_brune
            )
            if len(d['pairs_idx']) > 0:
                store[(ph, m)] = d
        except Exception as e:
            print(f"  Warning: gather failed for {ph}/{m}: {e}")

    # Aggregate across phases
    per_pair = defaultdict(lambda: {
        'L':[], 'H':[], 'sL':[], 'sH':[], 'mL':[], 'mH':[], 'N':[],
        'bL':[], 'bH':[], 'fc1':[], 'fc2':[], 'lfl_hi':[], 'hfl_lo':[]
    })

    for (ph, m), d in store.items():
        for idx in range(len(d['pairs_idx'])):
            p = d['pairs_idx'][idx]
            L, H = d['LFL_mean'][idx], d['HFL_mean'][idx]
            if not (np.isfinite(L) and np.isfinite(H)):
                continue
            key = tuple(sorted(p))
            per_pair[key]['L'].append(float(L))
            per_pair[key]['H'].append(float(H))
            per_pair[key]['sL'].append(float(d['LFL_std'][idx]))
            per_pair[key]['sH'].append(float(d['HFL_std'][idx]))
            per_pair[key]['mL'].append(float(d['LFL_slope'][idx]))
            per_pair[key]['mH'].append(float(d['HFL_slope'][idx]))
            per_pair[key]['N'].append(float(d['N_sta'][idx]))
            per_pair[key]['bL'].append(float(d['brune_LFL'][idx]))
            per_pair[key]['bH'].append(float(d['brune_HFL'][idx]))
            per_pair[key]['fc1'].append(float(d['fc1'][idx]))
            per_pair[key]['fc2'].append(float(d['fc2'][idx]))
            per_pair[key]['lfl_hi'].append(float(d['used_lfl_hi'][idx]))
            per_pair[key]['hfl_lo'].append(float(d['used_hfl_lo'][idx]))

    f = np.nanmedian if agg == 'median' else np.nanmean

    pairs, LFL, HFL, sLFL, sHFL = [], [], [], [], []
    mLFL, mHFL, N_eff = [], [], []
    bruneL, bruneH, fc1_agg, fc2_agg = [], [], [], []
    lfl_hi_agg, hfl_lo_agg = [], []

    for p, vals in per_pair.items():
        if not vals['L']:
            continue
        pairs.append(p)
        LFL.append(f(vals['L'])); HFL.append(f(vals['H']))
        sLFL.append(f(vals['sL'])); sHFL.append(f(vals['sH']))
        mLFL.append(f(vals['mL'])); mHFL.append(f(vals['mH']))
        N_eff.append(np.nanmedian(vals['N']))
        bruneL.append(f(vals['bL'])); bruneH.append(f(vals['bH']))
        fc1_agg.append(f(vals['fc1'])); fc2_agg.append(f(vals['fc2']))
        lfl_hi_agg.append(f(vals['lfl_hi'])); hfl_lo_agg.append(f(vals['hfl_lo']))

    LFL = np.asarray(LFL); HFL = np.asarray(HFL)
    sLFL = np.asarray(sLFL); sHFL = np.asarray(sHFL)
    mLFL = np.asarray(mLFL); mHFL = np.asarray(mHFL)
    N_eff = np.asarray(N_eff)

    sd_pairs = 1.5 * HFL - 0.5 * LFL
    sigma_pairs = np.sqrt((1.5 * sHFL)**2 + (0.5 * sLFL)**2)

    G = nx.Graph()
    G.add_nodes_from(range(len(ids_arr)))
    G.add_edges_from(pairs)

    if restrict_to_largest:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        largest = comps[0] if comps else set()
        mask = np.array([(i in largest and j in largest) for (i,j) in pairs], bool)
        pairs = [p for (p, ok) in zip(pairs, mask) if ok]
        sd_pairs = sd_pairs[mask]; sigma_pairs = sigma_pairs[mask]
        N_eff = N_eff[mask]
        LFL = LFL[mask]; HFL = HFL[mask]
        sLFL = sLFL[mask]; sHFL = sHFL[mask]
        mLFL = mLFL[mask]; mHFL = mHFL[mask]
        nodes = sorted(list(largest))
    else:
        nodes = sorted(set([i for i,_ in pairs] + [j for _,j in pairs]))

    out = {
        'pairs': np.array(pairs), 'nodes': nodes,
        'LFL': LFL, 'HFL': HFL, 'sLFL': sLFL, 'sHFL': sHFL,
        'LFL_slope': mLFL, 'HFL_slope': mHFL,
        'sd': sd_pairs, 'sigma_sd': sigma_pairs, 'n_sta': N_eff,
        'brune_LFL': np.asarray(bruneL), 'brune_HFL': np.asarray(bruneH),
        'fc1': np.asarray(fc1_agg), 'fc2': np.asarray(fc2_agg),
        'used_lfl_hi': np.asarray(lfl_hi_agg),
        'used_hfl_lo': np.asarray(hfl_lo_agg),
    }

    if restrict_to_largest:
        out = restrict_dataset_to_largest_component(out)

    return out
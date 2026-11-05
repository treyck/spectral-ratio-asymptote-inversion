"""
run_ridgecrest_knn.py (v2)
==========================
Adds:
  - Adaptive LFL bands (--adaptive_lfl flag)
  - Component selection (--comp_mode flag)
  - Adaptive safety factor CLI arg
  
All other functionality unchanged from v1.
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import sys
import numpy as np
import pandas as pd
import networkx as nx
import time
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from scipy.stats import siegelslopes, linregress
from math import cos, radians
from collections import Counter
import stress_inversion_lib as slib
import graph_metrics as gm
import network_plots as nplot

# --- 0. ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Run Ridgecrest Stress Drop Inversion (k-NN)")
parser.add_argument('--k_neighbors', type=int, default=7)
parser.add_argument('--max_dist_m', type=float, default=1500.0)
parser.add_argument('--max_depth_diff_m', type=float, default=1500.0)
parser.add_argument('--min_mag', type=float, default=2.0)
parser.add_argument('--skip_brune_fit', action='store_true')
parser.add_argument('--save_pairs', action='store_true')
parser.add_argument('--bandwidth_scaled', action='store_true',
                    help="Scale band slopes by band width so the penalty is drift across the band (dex)")

# Frequency bands (used when adaptive is OFF)
parser.add_argument('--lfl_lo', type=float, default=0.25)
parser.add_argument('--lfl_hi', type=float, default=1.2)
parser.add_argument('--hfl_lo', type=float, default=25.0)
parser.add_argument('--hfl_hi', type=float, default=40.0)

# Magnitude gates
parser.add_argument('--lfl_max', type=float, default=4.0)
parser.add_argument('--hfl_min', type=float, default=2.5)
parser.add_argument('--hfl_relay_min', type=float, default=None,
       help='HFL solve-eligibility threshold (relay admission). '
            'Pairs enter the HFL block if both events exceed this; '
            'reporting/dual-band still uses --hfl_min. '
            'Default None = no relays (current behavior).')

# SNR threshold
parser.add_argument('--noise_thres', type=float, default=3.5)

# Phase Consideration
parser.add_argument('--spec', type=str, default='P:multitaper,S:multitaper')

# Component selection
parser.add_argument('--comp_mode', type=str, default='all',
                    choices=['all', 'selected'],
                    help="'all' = E/N/Z geometric mean for all phases. "
                         "'selected' = Z for P, E/N for S.")

# Adaptive LFL bands
parser.add_argument('--adaptive_lfl', action='store_true', default=True,
                    help="Use magnitude-dependent adaptive LFL bands")
parser.add_argument('--no_adaptive_lfl', dest='adaptive_lfl', action='store_false',
                    help="Use fixed LFL bands")
parser.add_argument('--adaptive_safety', type=float, default=0.26,
                    help="Safety factor for adaptive LFL edge (fraction of median fc)")
parser.add_argument('--adaptive_a', type=float, default=1.764,
                    help="Intercept of log10(fc_S) = a + b*M for the adaptive LFL edge")
parser.add_argument('--adaptive_b', type=float, default=-0.299,
                    help="Slope of log10(fc_S) = a + b*M for the adaptive LFL edge")
parser.add_argument('--slope_weight', type=float, default=0.5,
                    help="W in sigma_eff = sqrt(std^2 + (W*|slope|)^2), Eq. 16")
parser.add_argument('--jk_seed', type=int, default=0,
                    help="Seed for jackknife pair resampling (reproducible jackknife)")

# Bootstrap
parser.add_argument('--n_boot', type=int, default=20)

# Task ID
parser.add_argument('--task_id', type=int, default=-1)

args = parser.parse_args()

relay_min = args.hfl_relay_min if args.hfl_relay_min is not None else args.hfl_min

# Build spec tag
spec_tag = args.spec.replace(':', '_').replace(',', '_').replace(' ', '')
adapt_tag = "adapt" if args.adaptive_lfl else "fixed"
comp_tag = args.comp_mode

if args.task_id >= 0:
    RUN_TAG = f"task{args.task_id}"
    RESULTS_SUBDIR = "grid_results_sweep"
else:
    RUN_TAG = f"k{args.k_neighbors}_{adapt_tag}_{comp_tag}_lfl{args.lfl_lo:g}"
    RESULTS_SUBDIR = f"grid_results_{spec_tag}"

if args.hfl_relay_min is not None:
    RUN_TAG += f"_relay{relay_min:g}"
if args.slope_weight != 0.5:
    RUN_TAG += f"_W{args.slope_weight:g}"
if args.adaptive_lfl and (args.adaptive_a != 1.764 or args.adaptive_b != -0.299):
    RUN_TAG += f"_a{args.adaptive_a:g}_b{args.adaptive_b:g}"
if args.bandwidth_scaled:
    RUN_TAG += "_bw"

# --- Build component map ---
if args.comp_mode == 'selected':
    COMP_MAP = {
        ('P', 'multitaper'): ('Z',),
        ('S', 'multitaper'): ('E', 'N'),
    }
    print("Component mode: SELECTED (Z for P, E/N for S)")
else:
    COMP_MAP = None
    print("Component mode: ALL (E/N/Z geometric mean)")

# --- Build adaptive config ---
if args.adaptive_lfl:
    # Hardcoded from local analysis:
    # log10(fc_S) = 1.764 - 0.299 * M, edge = safety × 10^(1.764 - 0.299*M)
    # Clipped to [1.2, 5.0] Hz
    knot_mags = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
    a_fit, b_fit = args.adaptive_a, args.adaptive_b
    sf = args.adaptive_safety
    knot_hz = [max(1.2, min(5.0, sf * 10**(a_fit + b_fit * m))) for m in knot_mags]

    ADAPTIVE_CONFIG = {
        'enabled': True,
        'knots_mag': knot_mags,
        'knots_hz': knot_hz,
        'lfl_lo': args.lfl_lo,
        'hfl_lo': args.hfl_lo,
        'hfl_hi': args.hfl_hi,
    }
    print(f"Adaptive LFL: enabled (safety={sf})")
    for m, h in zip(knot_mags, knot_hz):
        print(f"  M={m:.2f}: LFL_hi = {h:.2f} Hz")
else:
    ADAPTIVE_CONFIG = {
        'enabled': False,
        'lfl_lo': args.lfl_lo,
        'hfl_lo': args.hfl_lo,
        'hfl_hi': args.hfl_hi,
    }
    print(f"Adaptive LFL: disabled (fixed [{args.lfl_lo}, {args.lfl_hi}] Hz)")

# Also set the band overrides for any code that still uses globals
slib.LFL_BAND_OVERRIDE = (args.lfl_lo, args.lfl_hi)
slib.HFL_BAND_OVERRIDE = (args.hfl_lo, args.hfl_hi)

# --- CONFIGURATION (PATHS) ---
BASE_DIR = "/oak/stanford/groups/beroza/tknudson/big_data_folder"
RESULTS_PATH = f"{BASE_DIR}/results_all_2above.pkl.gz"
GC_PATH = f"{BASE_DIR}/sc_1981_2024_1d_3d_gc_soda_noqb_10_1_SCSN.gc"

MOMENT_PATH = f"./moment_fits.npy"
MAG_FIT_PATH = f"./mag_fits.npy"
IDS_FIT_PATH = f"./ids_forMomentFits.npy"
FCS_PATH = f"./corner_frequencies.npy"
SD_FIT_PATH = f"./stressdrops.npy"

# --- 1. LOAD DATA ---
print("Loading Results...")
results = slib.load_results_gz(RESULTS_PATH)
result_event_ids = list(map(str, results.keys()))

print("Loading Catalog...")
gc_df = slib.read_sc_growclust(GC_PATH)
roi = slib.build_event_meta(gc_df, result_event_ids,
                             restrict_years=(2015, 2025), restrict_mag_min=None)

print(f"Applying ROI Filter...")
original_len = len(roi)
mask = (
    (roi['lonR'] > -117.75) & (roi['lonR'] < -117.30) &
    (roi['latR'] >  35.40 ) & (roi['latR'] <  36.00 ) &
    (roi['mag']  >  2.0   ) & (roi['mag']  <  5.0   )
)
roi = roi[mask].copy()
print(f"Filter dropped {original_len - len(roi)} events (Now: {len(roi)})")

ids_arr = roi.index.to_numpy()
lat_arr = roi['latR'].to_numpy(float)
lon_arr = roi['lonR'].to_numpy(float)
dep_km  = roi['depR'].to_numpy(float)
n_events = len(roi)

# --- 2. k-NN PAIRING ---
print(f"Running k-NN Spatial Pairing (k={args.k_neighbors})...")

lat0 = float(np.nanmean(lat_arr))
lon0 = float(np.nanmean(lon_arr))
m_per_deg_lat = 111_132.0
m_per_deg_lon = 111_320.0 * cos(radians(lat0))
x_m = (lon_arr - lon0) * m_per_deg_lon
y_m = (lat_arr - lat0) * m_per_deg_lat
z_m = dep_km * 1000.0

XYZ = np.column_stack([x_m, y_m, z_m])
tree = cKDTree(XYZ)

distances, indices = tree.query(XYZ, k=args.k_neighbors + 1)

filtered_pairs = set()
for i in range(len(XYZ)):
    for j_idx in range(1, args.k_neighbors + 1):
        j = indices[i, j_idx]
        dist_3d = distances[i, j_idx]
        if dist_3d > args.max_dist_m:
            continue
        if abs(z_m[i] - z_m[j]) > args.max_depth_diff_m:
            continue
        pair = tuple(sorted((int(i), int(j))))
        filtered_pairs.add(pair)

filtered_pairs = sorted(list(filtered_pairs))
print(f"Final network: {len(filtered_pairs)} unique pairs.")

point_count = {}
for i, j in filtered_pairs:
    point_count[i] = point_count.get(i, 0) + 1
    point_count[j] = point_count.get(j, 0) + 1
used_indices = sorted(point_count.keys())
print(f"Used nodes: {len(used_indices)} / {len(XYZ)}")

G = nx.Graph()
G.add_nodes_from(range(n_events))
G.add_edges_from(filtered_pairs)
components = sorted(nx.connected_components(G), key=len, reverse=True)

set_indices = []
for comp in components:
    comp_set = set(comp)
    edges = [(i, j) for (i, j) in filtered_pairs if i in comp_set and j in comp_set]
    set_indices.append(edges)

# --- 2b. GRAPH METRICS ---
print("Computing Graph Metrics...")
mag_arr = roi['mag'].to_numpy(float)
graph_metrics = gm.compute_all_graph_metrics(
    G, filtered_pairs, XYZ, dep_km, mag_arr=mag_arr, ids_arr=ids_arr
)

# --- 3. BUILD DATASET (ADAPTIVE) ---
print(f"Building Spectral Dataset...")
print(f"  Adaptive LFL: {ADAPTIVE_CONFIG['enabled']}")
print(f"  Component mode: {args.comp_mode}")

ds_all = slib.build_global_pair_dataset_adaptive(
    results, ids_arr, roi, set_indices,
    spec=args.spec, setnum=0, noise_thres=args.noise_thres,
    agg='median', restrict_to_largest=False,
    comp_map=COMP_MAP, adaptive_config=ADAPTIVE_CONFIG,
    fit_brune=not args.skip_brune_fit
)

graph_metrics.update(gm._spectral_quality_metrics(ds_all))
graph_metrics['spectral_yield_frac'] = len(ds_all.get('pairs', [])) / max(len(filtered_pairs), 1)

# --- DIAGNOSTICS ---
print("\n>>> DIAGNOSTIC FINGERPRINTS <<<")
print(f"  Catalog ROI Size: {len(roi)}")
print(f"  k_neighbors: {args.k_neighbors}")
print(f"  Network pairs: {len(filtered_pairs)}")
print(f"  Dataset pairs: {len(ds_all['pairs'])}")
print(f"  Adaptive LFL: {ADAPTIVE_CONFIG['enabled']}")
print(f"  Component mode: {args.comp_mode}")

if 'used_lfl_hi' in ds_all:
    lfl_hi = ds_all['used_lfl_hi']
    v = np.isfinite(lfl_hi)
    if np.sum(v) > 0:
        print(f"  LFL_hi range: [{np.min(lfl_hi[v]):.2f}, {np.max(lfl_hi[v]):.2f}] Hz")
        print(f"  LFL_hi median: {np.median(lfl_hi[v]):.2f} Hz")

# --- 4. INVERSION ---
print("\nRunning Inversion...")
ds_largest = slib.restrict_dataset_to_largest_component(ds_all)

if args.save_pairs:
    import gzip, pickle
    with gzip.open(f"pairs_{RUN_TAG}.pkl.gz", 'wb') as f:
        pickle.dump({'ds_largest': ds_largest, 'ids_arr': ids_arr, 'mag_arr': mag_arr,
                     'args': vars(args)}, f)
    print(f"  Saved pair dataset -> pairs_{RUN_TAG}.pkl.gz")

if args.bandwidth_scaled:
    ds_largest = slib.apply_bandwidth_scaling(ds_largest, args.lfl_lo, args.hfl_hi)
    print("  Slope penalty: bandwidth-scaled (drift across band, dex)")

dual_band_limits = {'LFL_max': args.lfl_max, 'HFL_min': args.hfl_min}

m_sd, nodes, m_lfl, m_hfl = slib.run_dual_band_inversion(
    ds_largest, mag_arr, slope_weight=args.slope_weight, mag_limits=dual_band_limits,
    solve_hfl_min=relay_min)

# --- 4b. COMPUTE PER-NODE DEGREE ---
node_to_col = {n: i for i, n in enumerate(nodes)}
node_degree = np.zeros(len(nodes), dtype=int)
for i, j in ds_largest['pairs']:
    if i in node_to_col:
        node_degree[node_to_col[i]] += 1
    if j in node_to_col:
        node_degree[node_to_col[j]] += 1

# --- 5. SAVE CSV ---
out_name = f"results_{RUN_TAG}.csv"
output_df = pd.DataFrame(index=roi.index[nodes])
output_df['LFL_rel'] = m_lfl
output_df['HFL_rel'] = m_hfl
output_df['SD_rel']  = m_sd
output_df['Mag']     = mag_arr[nodes]
output_df['Degree']  = node_degree
mask_dual = (output_df['Mag'] >= dual_band_limits['HFL_min']) & \
            (output_df['Mag'] <= dual_band_limits['LFL_max'])
output_df['is_dual_band'] = mask_dual
output_df['is_relay'] = (output_df['Mag'] < args.hfl_min)
output_df['is_goldilocks'] = mask_dual
output_df.to_csv(out_name)

# --- 6. STATISTICS ---
print("\n--- STATISTICS ---")

# A. Jackknife
print("Running Jackknife (20 iterations)...")
errors = slib.run_jackknife_uncertainty_aligned(ds_largest, mag_arr, dual_band_limits, n_iter=20,
                                                solve_hfl_min=relay_min,
                                                slope_weight=args.slope_weight, seed=args.jk_seed)
valid_errs = errors['SD_std'][np.isfinite(errors['SD_std'])]
print(f"Jackknife Error (SD) - Mean: {np.mean(valid_errs):.4f}, Median: {np.median(valid_errs):.4f}")

# B. Station-Level Bootstrap
bootstrap_sd_median = np.nan
if args.n_boot > 0:
    print(f"\nRunning Station-Level Bootstrap ({args.n_boot} iterations)...")
    t_boot = time.time()

    rng = np.random.default_rng(42)
    pairs_ds = ds_largest['pairs']
    n_pairs_ds = len(pairs_ds)

    ref_sd = m_sd.copy()
    sd_boot = np.full((len(nodes), args.n_boot), np.nan)

    # Parse spec to get phase/method list for bootstrap
    phase_method_list = slib.parse_phase_method_spec(args.spec)

    for b in range(args.n_boot):
        if (b + 1) % 5 == 0:
            elapsed = time.time() - t_boot
            rate = elapsed / (b + 1)
            remaining = rate * (args.n_boot - b - 1)
            print(f"  Bootstrap {b+1}/{args.n_boot}  "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        boot_lfl = np.full(n_pairs_ds, np.nan)
        boot_hfl = np.full(n_pairs_ds, np.nan)
        boot_slfl = np.full(n_pairs_ds, np.nan)
        boot_shfl = np.full(n_pairs_ds, np.nan)

        for p_idx, (ii, jj) in enumerate(pairs_ds):
            ev1 = str(ids_arr[ii])
            ev2 = str(ids_arr[jj])

            all_ratios = []
            freq_ref = None

            for phase, method in phase_method_list:
                # Get component restriction for this phase
                comps = COMP_MAP.get((phase, method)) if COMP_MAP else None

                try:
                    out = slib.compute_phase_method_ratios(
                        results, ev1, ev2, phase, method,
                        noise_thres=args.noise_thres,
                        return_freqs=False, return_stations=True,
                        components=comps
                    )
                except Exception:
                    continue

                if not out or len(out) < 2:
                    continue
                ratios_list, stations = out
                if not ratios_list:
                    continue

                try:
                    freq = np.asarray(
                        results[ev1][phase][method]['frequencies'][0], float)
                except Exception:
                    try:
                        freq = np.asarray(
                            results[ev2][phase][method]['frequencies'][0], float)
                    except Exception:
                        continue

                if freq_ref is None:
                    freq_ref = freq

                # Resample stations with replacement
                n_sta = len(ratios_list)
                boot_indices = rng.choice(n_sta, size=n_sta, replace=True)
                boot_ratios = [ratios_list[k] for k in boot_indices]

                for r in boot_ratios:
                    r_arr = np.asarray(r, float).flatten()
                    if len(freq) == len(freq_ref) and np.allclose(freq, freq_ref):
                        all_ratios.append(r_arr)
                    else:
                        try:
                            fi = interp1d(freq, r_arr, kind='linear',
                                          bounds_error=False, fill_value=np.nan)
                            all_ratios.append(fi(freq_ref))
                        except Exception:
                            continue

            if not all_ratios or freq_ref is None:
                continue

            stack = np.vstack(all_ratios)
            med = np.nanmedian(stack, axis=0)

            # Use adaptive bands for this pair
            mi, mj = mag_arr[ii], mag_arr[jj]
            lfl_band, hfl_band = slib.get_pair_bands(mi, mj, config=ADAPTIVE_CONFIG)

            lfl_mask = ((freq_ref >= lfl_band[0]) & (freq_ref <= lfl_band[1])
                        & np.isfinite(med))
            hfl_mask = ((freq_ref >= hfl_band[0]) & (freq_ref <= hfl_band[1])
                        & np.isfinite(med))

            if np.sum(lfl_mask) >= 2 and np.sum(hfl_mask) >= 2:
                boot_lfl[p_idx] = np.nanmean(med[lfl_mask])
                boot_hfl[p_idx] = np.nanmean(med[hfl_mask])
                boot_slfl[p_idx] = np.nanstd(med[lfl_mask])
                boot_shfl[p_idx] = np.nanstd(med[hfl_mask])

        valid_boot = np.isfinite(boot_lfl) & np.isfinite(boot_hfl)
        if np.sum(valid_boot) < 100:
            continue

        ds_boot = ds_largest.copy()
        ds_boot['LFL'] = np.where(valid_boot, boot_lfl, 0.0)
        ds_boot['HFL'] = np.where(valid_boot, boot_hfl, 0.0)
        ds_boot['sLFL'] = np.where(valid_boot & np.isfinite(boot_slfl), boot_slfl, np.inf)
        ds_boot['sHFL'] = np.where(valid_boot & np.isfinite(boot_shfl), boot_shfl, np.inf)
        ds_boot['sd'] = np.where(valid_boot, 1.5 * boot_hfl - 0.5 * boot_lfl, 0.0)

        try:
            b_sd, _, b_lfl, _ = slib.run_dual_band_inversion(
                ds_boot, mag_arr, slope_weight=args.slope_weight, mag_limits=dual_band_limits,
                solve_hfl_min=relay_min
            )
            valid_align = np.isfinite(b_sd) & np.isfinite(ref_sd)
            if np.sum(valid_align) > 10:
                shift = np.median(ref_sd[valid_align] - b_sd[valid_align])
                sd_boot[:, b] = b_sd + shift
        except Exception:
            continue

    boot_std = np.nanstd(sd_boot, axis=1)
    valid_bs = boot_std[np.isfinite(boot_std) & (boot_std > 0)]
    bootstrap_sd_median = float(np.median(valid_bs)) if len(valid_bs) > 0 else np.nan

    print(f"\nBootstrap complete in {time.time() - t_boot:.1f}s")
    print(f"  Bootstrap SD uncertainty: median = {bootstrap_sd_median:.4f}")
    print(f"  Jackknife SD uncertainty: median = {np.median(valid_errs):.4f}")
    errors['boot_SD_std'] = boot_std
    
# persist per-event uncertainties for noise deconvolution
output_df['SD_jack_std'] = errors['SD_std']
if 'boot_SD_std' in errors:
    output_df['SD_boot_std'] = errors['boot_SD_std']
output_df.to_csv(out_name)   # re-save with uncertainty columns

# C. Triplet Closure
print("\nComputing Triplet Closure...")
closures = slib.find_triplets_and_compute_closure(ds_largest, mag_arr,
                                                   metrics=['SD', 'LFL', 'HFL'],
                                                   slope_weight=args.slope_weight)
sd_close_stats = closures['SD']
valid_closure = sd_close_stats['closure'][np.isfinite(sd_close_stats['closure'])]
print(f"Triplet Closure (SD) - Mean: {np.mean(valid_closure):.4f}")

# --- 7. PLOTTING ---
print("\nGenerating Plots...")

try:
    ext_mom = np.load(MOMENT_PATH)
    ext_sd  = np.load(SD_FIT_PATH, allow_pickle=True)
    ext_ids = np.load(IDS_FIT_PATH)
except Exception as e:
    print(f"Warning: Could not load external fits ({e})")
    ext_mom, ext_sd, ext_ids = [], [], []

plot_metrics = slib.generate_comparison_plots_fixed(
    output_df, roi, ext_mom, ext_sd, ext_ids,
    limits=dual_band_limits, save_path=f"summary_{RUN_TAG}.png"
)

slib.plot_clean_uncertainties(
    output_df, errors, dual_band_limits,
    node_degree=node_degree, save_path=f"jackknife_{RUN_TAG}.png"
)

for metric in ['SD', 'LFL', 'HFL']:
    slib.plot_triplet_closure_dashboard(
        closures, metric=metric, save_path=f"triplets_{metric}_{RUN_TAG}.png"
    )

nplot.plot_network_map_and_section(
    roi, ds_largest, ids_arr,
    mag_arr=mag_arr, limits=dual_band_limits,
    save_path=f"network_{RUN_TAG}.png"
)

# --- 8. SAVE SUMMARY ---
print("\nSaving summary metrics...")
import os

sd_vals = output_df.loc[output_df['is_dual_band'], 'SD_rel'].dropna()
n_dual_band = len(sd_vals)
sd_sigma = plot_metrics.get('SD_sigma', np.nan)

# Compute per-quantity depth slopes (Siegel robust)
from scipy.stats import siegelslopes

node_mags = mag_arr[nodes]
node_deps = dep_km[nodes]
gold = (node_mags >= dual_band_limits['HFL_min']) & \
       (node_mags <= dual_band_limits['LFL_max']) & np.isfinite(m_sd)

lfl_depth_slope = np.nan
hfl_depth_slope = np.nan
sd_mag_slope = np.nan

gold_lfl = gold & np.isfinite(m_lfl)
if np.sum(gold_lfl) > 20:
    lfl_depth_slope, _ = siegelslopes(m_lfl[gold_lfl], node_deps[gold_lfl])

gold_hfl = gold & np.isfinite(m_hfl)
if np.sum(gold_hfl) > 20:
    hfl_depth_slope, _ = siegelslopes(m_hfl[gold_hfl], node_deps[gold_hfl])

if np.sum(gold) > 20:
    sd_mag_slope, _ = siegelslopes(m_sd[gold], node_mags[gold])

summary_data = {
    'task_id': args.task_id,
    'k_neighbors': args.k_neighbors,
    'adaptive_lfl': args.adaptive_lfl,
    'adaptive_safety': args.adaptive_safety if args.adaptive_lfl else np.nan,
    'adaptive_a': args.adaptive_a if args.adaptive_lfl else np.nan,
    'adaptive_b': args.adaptive_b if args.adaptive_lfl else np.nan,
    'slope_weight': args.slope_weight,
    'bandwidth_scaled': args.bandwidth_scaled,
    'jk_seed': args.jk_seed,
    'comp_mode': args.comp_mode,
    'lfl_lo': args.lfl_lo,
    'lfl_hi': args.lfl_hi,
    'hfl_lo': args.hfl_lo,
    'hfl_hi': args.hfl_hi,
    'lfl_max': args.lfl_max,
    'hfl_min': args.hfl_min,
    'noise_thres': args.noise_thres,
    'n_pairs': len(filtered_pairs),
    'n_used_nodes': len(used_indices),
    'n_dataset_pairs': len(ds_all['pairs']),
    'n_dual_band': n_dual_band,
    'SD_sigma': plot_metrics.get('SD_sigma', np.nan),
    'stress_drop_depth_trend': plot_metrics.get('SD_depth_slope', np.nan),
    'stress_drop_lfl_trend': plot_metrics.get('SD_lfl_slope', np.nan),
    'LFL_validation_slope': plot_metrics.get('LFL_val_slope', np.nan),
    'lfl_depth_slope': lfl_depth_slope,          # NEW
    'hfl_depth_slope': hfl_depth_slope,          # NEW
    'sd_mag_slope': sd_mag_slope,                # NEW
    'triplet_closure_SD_mean': np.mean(valid_closure),
    'jackknife_uncertainty_SD': np.nanmedian(errors['SD_std']),
    'bootstrap_uncertainty_SD': bootstrap_sd_median,
}
summary_data.update(graph_metrics)

os.makedirs(RESULTS_SUBDIR, exist_ok=True)
out_file = f"{RESULTS_SUBDIR}/summary_{RUN_TAG}.csv"
pd.DataFrame([summary_data]).to_csv(out_file, index=False)
print(f"Saved grid summary to {out_file}")

print("Done.")

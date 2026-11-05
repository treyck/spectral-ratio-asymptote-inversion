# spectral-ratio-asymptote-inversion

Inversion code for **Knudson, T. C., Shaw, B. E., Ellsworth, W. L., and Beroza, G. C.,
*Stress Drops Without Fitting Corner Frequencies: A Graph Inversion of Spectral
Ratio Asymptotes for the Ridgecrest Earthquake Sequence*, submitted to
JGR: Solid Earth (2026).**

The method measures the low- and high-frequency asymptotes of event-pair
spectral ratios, links thousands of pairs in a k-nearest-neighbor graph, and
solves for relative low- and high-frequency levels, and hence relative stress
drops, in one weighted least-squares inversion. This repository contains the
graph construction, band measurements, weighting, inversion, and the bootstrap
and jackknife uncertainty analysis that produced the paper's results. The
spectra are computed by a separate package (see Inputs). Figure scripts are not
included.

The stress-drop catalog produced by the paper is archived at Zenodo:
https://doi.org/10.5281/zenodo.23726995

## Contents

```
inversion/
  run_ridgecrest_knn_v2.py     driver: builds the graph, runs the inversion, bootstrap, jackknife
  stress_inversion_lib.py      library: spectral ratios, band measurements, weights, inversion, closure
  graph_metrics.py             graph statistics used by the driver
  network_plots.py             graph plotting utilities imported by the driver
  organize_sweep_v2.py         sorts sweep outputs into per-phase folders
  slurm/                       the job scripts used for the preferred run and the k_nn sweep
requirements.txt
README.md
```

## Inputs

| Input | Source |
|---|---|
| Event spectra, `results_all_2above.pkl.gz` (P, S, and noise multitaper spectra per event and station) | computed with https://github.com/treyck/multiphase_multimethod_spec |
| SCEDC relocated catalog, `sc_1981_2024_1d_3d_gc_soda_noqb_10_1_SCSN.gc` (updated 7 March 2025) | https://scedc.caltech.edu/data/alt-2011-dd-hauksson-yang-shearer.html (Hauksson et al., 2012; GrowClust extension) |

Waveforms and parametric data are from the Caltech/USGS Southern California
Seismic Network (doi:10.7914/SN/CI) via the Southern California Earthquake Data
Center (doi:10.7909/C3WD3xH1). The catalog file is not redistributed here.

## Environment

Python 3.12. Install with `pip install -r requirements.txt` (numpy, scipy,
pandas, networkx, matplotlib, obspy). The spectra pickle was written under
NumPy 2; loading it under NumPy 1 requires remapping the `numpy._core` module
names on unpickling.

## Running

The paths at the top of `run_ridgecrest_knn_v2.py` (`BASE_DIR`, `RESULTS_PATH`,
`GC_PATH`) point to the authors' cluster and must be edited.

**Preferred inversion** (the paper's configuration), from `inversion/slurm/`:

```
python run_ridgecrest_knn_v2.py --k_neighbors 8 --max_dist_m 1500 --max_depth_diff_m 1500 \
  --min_mag 2.0 --lfl_lo 0.25 --lfl_hi 1.2 --hfl_lo 25 --hfl_hi 40 --lfl_max 4.0 --hfl_min 2.5 \
  --noise_thres 3.5 --hfl_relay_min 2.3 --adaptive_lfl --adaptive_safety 0.26 \
  --comp_mode all --spec "P:multitaper,S:multitaper" --slope_weight 0.25 --bandwidth_scaled \
  --n_boot 50 --save_pairs --skip_brune_fit
```

Outputs: a per-event results CSV (`results_k8_...csv`) with relative LFL, HFL,
and stress drop plus bootstrap and jackknife uncertainties; a pairs pickle
(`pairs_k8_...pkl.gz`) with the graph, per-pair measurements, and arguments;
and a one-row summary. The 3-D separation cap makes `--max_depth_diff_m`
redundant when the two values are equal.

**k_nn sweep** (Figures 6 and 12 of the paper): the same arguments with
`--k_neighbors` in {6, 7, 8, 9, 10, 12, 14, 16, 18, 20}, and for the
far-separation runs `--max_dist_m 3000` with k in {20, 40, 80}, each for
P-only, S-only, and P+S via `--spec`. The sweep job script submits these by
`--task_id`; `organize_sweep_v2.py` sorts the outputs into
`sweep_<tag>_{P,S,PS}` and `farsep_<tag>_{P,S,PS}` folders.

## Citation

If you use this code, please cite the paper above and this repository
(Zenodo DOI to be added on release).

## License

MIT; see `LICENSE`.

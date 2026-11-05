"""
graph_metrics.py
Computes graph-topology and eGF-quality metrics for the event pairing network.

All functions return flat dictionaries suitable for direct inclusion
in a summary CSV row.  Heavy operations (diameter, curvature) use
approximations so the full suite runs in seconds even for ~50k edges.
"""

import numpy as np
import networkx as nx
from collections import Counter


# ============================================================
# 1.  MASTER FUNCTION  (call this from run_ridgecrest.py)
# ============================================================

def compute_all_graph_metrics(G, filtered_pairs, XYZ, dep_km, mag_arr,
                              ids_arr=None, ds_all=None):
    """
    Parameters
    ----------
    G              : nx.Graph built from filtered_pairs
    filtered_pairs : list of (i, j) index tuples
    XYZ            : (N, 3) array of [x_m, y_m, z_m] used for the tree
    dep_km         : (N,) depth array in km
    mag_arr        : (N,) magnitude array (full ROI indexing)
    ids_arr        : (N,) event-id array  (optional, unused for now)
    ds_all         : spectral dataset dict (optional – for spectral-quality metrics)

    Returns
    -------
    dict : flat key-value pairs, all scalars or short strings
    """
    metrics = {}

    pairs_arr = np.array(filtered_pairs)  # (P, 2)

    # --- A. Basic topology ---
    metrics.update(_degree_metrics(G))

    # --- B. Edge-length distribution (3D Euclidean) ---
    metrics.update(_edge_distance_metrics(pairs_arr, XYZ))

    # --- C. Depth-difference distribution across edges ---
    metrics.update(_depth_diff_metrics(pairs_arr, dep_km))

    # --- D. Magnitude-difference distribution (eGF relevance) ---
    metrics.update(_mag_diff_metrics(pairs_arr, mag_arr))

    # --- E. Assortativity (degree & magnitude) ---
    metrics.update(_assortativity_metrics(G, mag_arr))

    # --- F. Clustering coefficient ---
    metrics.update(_clustering_metrics(G))

    # --- G. Connected-component stats ---
    metrics.update(_component_metrics(G))

    # --- H. Approximate diameter (largest component) ---
    metrics.update(_diameter_metrics(G))

    # --- I. Forman-Ricci curvature distribution ---
    metrics.update(_forman_curvature_metrics(G))

    # --- J. Spectral dataset quality (if available) ---
    if ds_all is not None:
        metrics.update(_spectral_quality_metrics(ds_all))

    return metrics


# ============================================================
# 2.  INDIVIDUAL METRIC FUNCTIONS
# ============================================================

def _degree_metrics(G):
    """Degree distribution summary."""
    degs = np.array([d for _, d in G.degree() if d > 0])
    if len(degs) == 0:
        return {k: np.nan for k in [
            'deg_mean', 'deg_median', 'deg_std', 'deg_min', 'deg_max',
            'deg_p10', 'deg_p90', 'n_isolated'
        ]}
    all_degs = np.array([d for _, d in G.degree()])
    return {
        'deg_mean':   float(np.mean(degs)),
        'deg_median': float(np.median(degs)),
        'deg_std':    float(np.std(degs)),
        'deg_min':    int(np.min(degs)),
        'deg_max':    int(np.max(degs)),
        'deg_p10':    float(np.percentile(degs, 10)),
        'deg_p90':    float(np.percentile(degs, 90)),
        'n_isolated': int(np.sum(all_degs == 0)),
    }


def _edge_distance_metrics(pairs_arr, XYZ):
    """3D Euclidean distance distribution across all edges."""
    if len(pairs_arr) == 0:
        return {k: np.nan for k in [
            'edge_dist3d_mean', 'edge_dist3d_median', 'edge_dist3d_std',
            'edge_dist3d_p10', 'edge_dist3d_p90',
            'edge_distH_mean', 'edge_distH_median',
        ]}
    dv = XYZ[pairs_arr[:, 0]] - XYZ[pairs_arr[:, 1]]
    dist3d = np.linalg.norm(dv, axis=1)
    distH = np.linalg.norm(dv[:, :2], axis=1)  # horizontal only
    return {
        'edge_dist3d_mean':   float(np.mean(dist3d)),
        'edge_dist3d_median': float(np.median(dist3d)),
        'edge_dist3d_std':    float(np.std(dist3d)),
        'edge_dist3d_p10':    float(np.percentile(dist3d, 10)),
        'edge_dist3d_p90':    float(np.percentile(dist3d, 90)),
        'edge_distH_mean':    float(np.mean(distH)),
        'edge_distH_median':  float(np.median(distH)),
    }


def _depth_diff_metrics(pairs_arr, dep_km):
    """Absolute depth difference across edges (km)."""
    if len(pairs_arr) == 0:
        return {k: np.nan for k in [
            'ddep_mean', 'ddep_median', 'ddep_std', 'ddep_p90'
        ]}
    ddep = np.abs(dep_km[pairs_arr[:, 0]] - dep_km[pairs_arr[:, 1]])
    return {
        'ddep_mean':   float(np.mean(ddep)),
        'ddep_median': float(np.median(ddep)),
        'ddep_std':    float(np.std(ddep)),
        'ddep_p90':    float(np.percentile(ddep, 90)),
    }


def _mag_diff_metrics(pairs_arr, mag_arr):
    """
    Magnitude difference across edges.
    eGF quality depends on having sufficient mag contrast AND
    keeping the smaller event close enough to act as a point source.
    """
    if len(pairs_arr) == 0:
        return {k: np.nan for k in [
            'dmag_mean', 'dmag_median', 'dmag_std', 'dmag_p10', 'dmag_p90',
            'frac_dmag_lt_0p5', 'frac_dmag_gt_1p0'
        ]}
    dmag = np.abs(mag_arr[pairs_arr[:, 0]] - mag_arr[pairs_arr[:, 1]])
    return {
        'dmag_mean':        float(np.mean(dmag)),
        'dmag_median':      float(np.median(dmag)),
        'dmag_std':         float(np.std(dmag)),
        'dmag_p10':         float(np.percentile(dmag, 10)),
        'dmag_p90':         float(np.percentile(dmag, 90)),
        'frac_dmag_lt_0p5': float(np.mean(dmag < 0.5)),   # low contrast (worse eGF)
        'frac_dmag_gt_1p0': float(np.mean(dmag > 1.0)),   # high contrast (better eGF)
    }


def _assortativity_metrics(G, mag_arr):
    """
    Degree assortativity: do high-degree nodes connect to high-degree nodes?
    Magnitude assortativity: do similar-magnitude events cluster together?
      (negative = good mixing; positive = magnitude-segregated graph)
    """
    out = {}
    try:
        out['degree_assortativity'] = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        out['degree_assortativity'] = np.nan

    # Magnitude assortativity via numeric attribute
    try:
        nx.set_node_attributes(G, {n: float(mag_arr[n]) for n in G.nodes()}, 'mag')
        out['mag_assortativity'] = float(
            nx.numeric_assortativity_coefficient(G, 'mag')
        )
    except Exception:
        out['mag_assortativity'] = np.nan

    return out


def _clustering_metrics(G):
    """
    Average clustering coefficient.
    High clustering = lots of triplets = better closure tests,
    but may also mean the graph is very local.
    """
    try:
        cc = nx.average_clustering(G)
        # Also compute transitivity (global clustering)
        trans = nx.transitivity(G)
        return {
            'avg_clustering_coeff': float(cc),
            'transitivity':         float(trans),
        }
    except Exception:
        return {'avg_clustering_coeff': np.nan, 'transitivity': np.nan}


def _component_metrics(G):
    """Connected component statistics."""
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    sizes = [len(c) for c in comps]
    return {
        'n_components':       len(comps),
        'largest_comp_size':  sizes[0] if sizes else 0,
        'largest_comp_frac':  float(sizes[0] / G.number_of_nodes()) if sizes and G.number_of_nodes() > 0 else 0.0,
        'second_comp_size':   sizes[1] if len(sizes) > 1 else 0,
    }


def _diameter_metrics(G, n_samples=50):
    """
    Approximate diameter of the largest connected component.
    Uses BFS from random source nodes to avoid the O(V*E) exact computation.
    """
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if not comps:
        return {'approx_diameter': np.nan, 'avg_shortest_path_sample': np.nan}

    largest = G.subgraph(comps[0]).copy()
    nodes_list = list(largest.nodes())

    if len(nodes_list) <= 500:
        # Small enough to compute exactly
        try:
            diam = nx.diameter(largest)
            avg_sp = nx.average_shortest_path_length(largest)
            return {'approx_diameter': int(diam), 'avg_shortest_path_sample': float(avg_sp)}
        except Exception:
            pass

    # Approximate: BFS from random sources, take max eccentricity
    rng = np.random.default_rng(42)
    sources = rng.choice(nodes_list, size=min(n_samples, len(nodes_list)), replace=False)
    max_ecc = 0
    path_lengths = []
    for src in sources:
        lengths = nx.single_source_shortest_path_length(largest, src)
        ecc = max(lengths.values())
        max_ecc = max(max_ecc, ecc)
        path_lengths.extend(lengths.values())

    return {
        'approx_diameter':          int(max_ecc),
        'avg_shortest_path_sample': float(np.mean(path_lengths)),
    }


def _forman_curvature_metrics(G):
    """
    Forman-Ricci curvature for unweighted graphs.
    For edge (u,v):  F(u,v) = 4 - deg(u) - deg(v)

    Negative curvature → edge bridges two hubs (bottleneck).
    Positive curvature → edge is in a sparse/leaf region.
    Near-zero → balanced local structure.
    
    For eGF: edges with very negative curvature may be 
    "bridge" pairs connecting different fault segments,
    which could degrade the eGF assumption.
    """
    if G.number_of_edges() == 0:
        return {k: np.nan for k in [
            'forman_mean', 'forman_median', 'forman_std',
            'forman_min', 'forman_p10', 'forman_p90',
            'frac_forman_negative'
        ]}

    deg = dict(G.degree())
    curvatures = np.array([4.0 - deg[u] - deg[v] for u, v in G.edges()])

    return {
        'forman_mean':           float(np.mean(curvatures)),
        'forman_median':         float(np.median(curvatures)),
        'forman_std':            float(np.std(curvatures)),
        'forman_min':            float(np.min(curvatures)),
        'forman_p10':            float(np.percentile(curvatures, 10)),
        'forman_p90':            float(np.percentile(curvatures, 90)),
        'frac_forman_negative':  float(np.mean(curvatures < 0)),
    }


def _spectral_quality_metrics(ds_all):
    """
    Metrics derived from the spectral dataset itself.
    Gives a sense of how 'clean' the spectral ratios are
    across the graph.
    """
    out = {}

    # Station coverage per pair
    n_sta = ds_all.get('n_sta', np.array([]))
    if len(n_sta) > 0:
        n_sta_clean = n_sta[np.isfinite(n_sta)]
        out['nsta_mean']   = float(np.mean(n_sta_clean)) if len(n_sta_clean) else np.nan
        out['nsta_median'] = float(np.median(n_sta_clean)) if len(n_sta_clean) else np.nan
        out['nsta_min']    = float(np.min(n_sta_clean)) if len(n_sta_clean) else np.nan
    else:
        out['nsta_mean'] = out['nsta_median'] = out['nsta_min'] = np.nan

    # LFL slope flatness (should be ~0 for good eGF)
    lfl_slope = ds_all.get('LFL_slope', np.array([]))
    if len(lfl_slope) > 0:
        valid = lfl_slope[np.isfinite(lfl_slope)]
        out['lfl_slope_abs_mean'] = float(np.mean(np.abs(valid))) if len(valid) else np.nan
        out['lfl_slope_abs_p90']  = float(np.percentile(np.abs(valid), 90)) if len(valid) else np.nan
    else:
        out['lfl_slope_abs_mean'] = out['lfl_slope_abs_p90'] = np.nan

    # HFL slope flatness
    hfl_slope = ds_all.get('HFL_slope', np.array([]))
    if len(hfl_slope) > 0:
        valid = hfl_slope[np.isfinite(hfl_slope)]
        out['hfl_slope_abs_mean'] = float(np.mean(np.abs(valid))) if len(valid) else np.nan
        out['hfl_slope_abs_p90']  = float(np.percentile(np.abs(valid), 90)) if len(valid) else np.nan
    else:
        out['hfl_slope_abs_mean'] = out['hfl_slope_abs_p90'] = np.nan

    # Intra-pair standard deviation (spectral noise proxy)
    slfl = ds_all.get('sLFL', np.array([]))
    shfl = ds_all.get('sHFL', np.array([]))
    if len(slfl) > 0:
        v = slfl[np.isfinite(slfl)]
        out['sLFL_mean'] = float(np.mean(v)) if len(v) else np.nan
    else:
        out['sLFL_mean'] = np.nan
    if len(shfl) > 0:
        v = shfl[np.isfinite(shfl)]
        out['sHFL_mean'] = float(np.mean(v)) if len(v) else np.nan
    else:
        out['sHFL_mean'] = np.nan

    # Fraction of pairs that survived spectral processing
    # (from total graph pairs to dataset pairs)
    out['spectral_yield'] = len(ds_all.get('pairs', [])) 

    return out

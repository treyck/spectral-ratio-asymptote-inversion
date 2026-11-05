"""
network_plots.py
================
Two-panel figure showing the event pair graph:
  Left:  Map view (lon vs lat)
  Right: Depth cross-section (lon vs depth)

Color convention (paper-wide):
  Blue (#4477CC)  = LFL-only events/edges
  Red  (#CC4422)  = HFL-only events/edges
  Green (#44AA77) = Dual-band events/edges
  Gray (#BBBBBB)  = Mixed edges
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec


# Paper-wide color constants
C_LFL   = '#4477CC'
C_HFL   = '#CC4422'
C_DUAL  = '#44AA77'
C_MIXED = '#BBBBBB'


def _classify_edges(pairs, mag_arr, limits):
    lfl_max = limits['LFL_max']
    hfl_min = limits['HFL_min']

    edge_types = {'dual': [], 'lfl': [], 'hfl': [], 'mixed': []}

    for idx, (i, j) in enumerate(pairs):
        mi = mag_arr[i] if i < len(mag_arr) else np.nan
        mj = mag_arr[j] if j < len(mag_arr) else np.nan

        i_dual = hfl_min <= mi <= lfl_max
        j_dual = hfl_min <= mj <= lfl_max
        i_lfl = mi <= lfl_max
        j_lfl = mj <= lfl_max
        i_hfl = mi >= hfl_min
        j_hfl = mj >= hfl_min

        if i_dual and j_dual:
            edge_types['dual'].append(idx)
        elif i_lfl and j_lfl and not (i_dual and j_dual):
            edge_types['lfl'].append(idx)
        elif i_hfl and j_hfl and not (i_dual and j_dual):
            edge_types['hfl'].append(idx)
        else:
            edge_types['mixed'].append(idx)

    return edge_types


def _classify_nodes(nodes, mag_arr, limits):
    lfl_max = limits['LFL_max']
    hfl_min = limits['HFL_min']

    dual_idx, lfl_idx, hfl_idx = [], [], []
    for pos, n in enumerate(nodes):
        m = mag_arr[n] if n < len(mag_arr) else np.nan
        if hfl_min <= m <= lfl_max:
            dual_idx.append(pos)
        elif m < hfl_min:
            lfl_idx.append(pos)
        elif m > lfl_max:
            hfl_idx.append(pos)

    return np.array(dual_idx), np.array(lfl_idx), np.array(hfl_idx)


def _build_segments(pairs, edge_indices, node_lons, node_y, node_to_pos):
    segments = []
    for idx in edge_indices:
        i, j = pairs[idx]
        if i not in node_to_pos or j not in node_to_pos:
            continue
        pi, pj = node_to_pos[i], node_to_pos[j]
        x1, x2 = node_lons[pi], node_lons[pj]
        y1, y2 = node_y[pi], node_y[pj]
        if np.isnan(x1) or np.isnan(x2) or np.isnan(y1) or np.isnan(y2):
            continue
        segments.append([(x1, y1), (x2, y2)])
    return segments


def plot_network_map_and_section(roi, ds_largest, ids_arr,
                                  output_df=None, mag_arr=None, limits=None,
                                  save_path=None):
    """
    Two-panel network figure: map view + depth cross-section.
    Events and edges colored by band membership.
    """
    pairs = ds_largest['pairs']
    nodes = ds_largest['nodes']

    if limits is None:
        limits = {'LFL_max': 4.0, 'HFL_min': 2.5}
    if mag_arr is None:
        mag_arr = roi['mag'].to_numpy(float)

    # Node positions
    node_lons = np.array([roi.loc[ids_arr[n], 'lonR'] if ids_arr[n] in roi.index
                          else np.nan for n in nodes])
    node_lats = np.array([roi.loc[ids_arr[n], 'latR'] if ids_arr[n] in roi.index
                          else np.nan for n in nodes])
    node_deps = np.array([roi.loc[ids_arr[n], 'depR'] if ids_arr[n] in roi.index
                          else np.nan for n in nodes])

    node_to_pos = {n: idx for idx, n in enumerate(nodes)}

    # Classify
    edge_types = _classify_edges(pairs, mag_arr, limits)
    dual_nodes, lfl_nodes, hfl_nodes = _classify_nodes(nodes, mag_arr, limits)

    n_dual  = len(edge_types['dual'])
    n_lfl   = len(edge_types['lfl'])
    n_hfl   = len(edge_types['hfl'])
    n_mixed = len(edge_types['mixed'])

    print(f"Edges: {n_dual} dual-band, {n_lfl} LFL-only, "
          f"{n_hfl} HFL-only, {n_mixed} mixed")
    print(f"Nodes: {len(dual_nodes)} dual-band, {len(lfl_nodes)} LFL-only, "
          f"{len(hfl_nodes)} HFL-only")

    # Edge draw order and style (back to front)
    edge_draw = [
        ('mixed', C_MIXED, 0.3, 0.16),
        ('lfl',   C_LFL,   0.4, 0.32),
        ('hfl',   C_HFL,   0.4, 0.32),
        ('dual',  C_DUAL,  0.5, 0.48),
    ]

    # Node draw order and style (back to front)
    node_draw = [
        (lfl_nodes,  C_LFL,  'LFL-only', 0.5, 5),
        (hfl_nodes,  C_HFL,  'HFL-only', 0.5, 5),
        (dual_nodes, C_DUAL, 'Dual-band', 0.7, 8),
    ]

    # ---- Figure with gridspec for proper aspect handling ----
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.12)

    ax_map = fig.add_subplot(gs[0])
    ax_sec = fig.add_subplot(gs[1])

    for ax, y_data, mode in [(ax_map, node_lats, 'map'),
                              (ax_sec, node_deps, 'section')]:

        # Draw edges
        for etype, color, lw, alpha in edge_draw:
            segs = _build_segments(pairs, edge_types[etype],
                                   node_lons, y_data, node_to_pos)
            if segs:
                lc = LineCollection(segs, colors=color,
                                    linewidths=lw, alpha=alpha, zorder=1)
                ax.add_collection(lc)

        # Draw nodes
        for node_idx, color, label, alpha, size in node_draw:
            if len(node_idx) > 0:
                ax.scatter(node_lons[node_idx], y_data[node_idx],
                           c=color, s=size, alpha=alpha,
                           edgecolor='k', linewidth=0.15, zorder=3)

        # Formatting
        ax.set_xlabel('Longitude', fontsize=11)
        ax.grid(True, ls=':', alpha=0.25)

        if mode == 'map':
            ax.set_ylabel('Latitude', fontsize=11)
            ax.set_aspect('equal')
        else:
            ax.set_ylabel('Depth (km)', fontsize=11)
            ax.invert_yaxis()

    # ---- Legend (on map panel) ----
    legend_els = [
        Line2D([0], [0], color=C_DUAL, lw=3, alpha=0.6,
               label=f'Dual-band edges ({n_dual})'),
        Line2D([0], [0], color=C_LFL, lw=3, alpha=0.6,
               label=f'LFL-only edges ({n_lfl})'),
        Line2D([0], [0], color=C_HFL, lw=3, alpha=0.6,
               label=f'HFL-only edges ({n_hfl})'),
        Line2D([0], [0], color=C_MIXED, lw=3, alpha=0.4,
               label=f'Mixed edges ({n_mixed})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_DUAL,
               markersize=7, markeredgecolor='k', markeredgewidth=0.3,
               label=f'Dual-band events ({len(dual_nodes)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_LFL,
               markersize=7, markeredgecolor='k', markeredgewidth=0.3,
               label=f'LFL-only events ({len(lfl_nodes)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_HFL,
               markersize=7, markeredgecolor='k', markeredgewidth=0.3,
               label=f'HFL-only events ({len(hfl_nodes)})'),
    ]
    ax_map.legend(handles=legend_els, fontsize=8, loc='lower left',
                  framealpha=0.95, edgecolor='#cccccc')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()
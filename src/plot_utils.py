import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap, to_rgb, rgb2hex
from scipy.ndimage import gaussian_filter1d as gf
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots
import pandas as pd


def interpolate_colors(color1, color2, num_colors=10):
    """
    Interpolate between two XKCD colors.

    Parameters:
        color1 (str): Starting color name (XKCD format).
        color2 (str): Ending color name (XKCD format).
        num_colors (int): Number of colors to generate.

    Returns:
        list: List of interpolated hex colors.
    """
    rgb1 = to_rgb(f'xkcd:{color1}')
    rgb2 = to_rgb(f'xkcd:{color2}')
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [rgb1, rgb2], N=num_colors)
    return [rgb2hex(cmap(i)) for i in np.linspace(0, 1, num_colors)]




def plot_local_average(data, behavioral_axis=None, arc_length_axis=None, top_colors=("black", "light gray"),
                       smooth_sigma=5, top_step=5, heatmap_scale='magma',
                       ba_title='Reaction Time (ms)', var_name='Variable', figsize=600,title=None):
    """
    Plots interpolated neural measures aligned to a behavioral axis, combining top traces and a square heatmap.

    Parameters:
        data (np.ndarray): Interpolated data array (arc-length points × behavioral points).
        behavioral_axis (np.ndarray): Behavioral measure axis.
        arc_length_axis (np.ndarray): Normalized arc-length axis values (default: 0–1).
        top_colors (tuple): Start and end color names for top traces.
        smooth_sigma (float): Gaussian smoothing sigma for visualization.
        top_step (int): Plot every nth trace in the top panel.
        heatmap_scale (str): Colormap for heatmap.
        ba_title (str): Title for behavioral axis (y-axis of heatmap).
        var_name (str): Label for heatmap colorbar (neural measure name).
        figsize (int): Pixel dimension for the figure size.

    Returns:
        go.Figure: Plotly figure object with two panels.
    """
    if arc_length_axis is None:
        arc_length_axis = np.linspace(0, 1, data.shape[0])

    if behavioral_axis is None:
        behavioral_axis = np.linspace(0, 1, data.shape[1])
        if ba_title == 'Reaction Time (ms)':
            ba_title = 'Normalized Reaction Time'

    smoothed_data = gf(data, sigma=smooth_sigma, axis=0) if smooth_sigma is not None else data
    colors = interpolate_colors(top_colors[0], top_colors[1], num_colors=len(behavioral_axis))

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.3, 0.7],
        shared_xaxes=True,
        vertical_spacing=0.02
    )

    # Top panel: Traces
    for idx in range(0, len(behavioral_axis), top_step):
        trace = smoothed_data[:, idx]
        hover_text = [f"{ba_title}: {behavioral_axis[idx]:.2f}" for _ in arc_length_axis]
        fig.add_trace(
            go.Scatter(
                x=arc_length_axis,
                y=trace,
                mode='lines',
                line=dict(color=colors[idx], width=2),
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False
            ),
            row=1, col=1
        )

    # Bottom panel: Heatmap (y-axis flipped)
    fig.add_trace(
        go.Heatmap(
            z=np.flipud(smoothed_data.T),
            x=arc_length_axis,
            y=behavioral_axis[::-1],
            colorscale=heatmap_scale,
            colorbar=dict(title=var_name)
        ),
        row=2, col=1
    )

    fig.update_xaxes(range=[arc_length_axis.min(), arc_length_axis.max()], row=2, col=1)
    fig.update_yaxes(range=[behavioral_axis.min(), behavioral_axis.max()], row=2, col=1, autorange=False)
    
    if title is None:
        title ='Local average of ' + var_name
        
    fig.update_layout(
        title=title,
        width=figsize,
        height=int(figsize * 1.2),
        margin=dict(l=100, r=100, t=100, b=100),
        xaxis2_title="Normalized Arc-Length",
        yaxis2_title=ba_title,
        yaxis1_title=var_name,
        showlegend=False
    )

    return fig



def plot_local_average_pop(manifolds, behavioral_axes, axis_names=['PC1 (Hz)', 'PC2 (Hz)', 'PC3 (Hz)'],
                           step=5, color_ranges=None, names=None, plot_size=1000,
                           title='Population Local Average', ba_title='Reaction Time (ms)'):
    """
    Plots multiple 3D submanifolds with hover information, customizable labels, and grouped legend.

    Parameters:
        manifolds (list[np.ndarray]): List of matrices (dim × arc-length × behavioral axis).
        behavioral_axes (list[np.ndarray]): List of arrays for behavioral axes.
        axis_names (list): Names of the 3 plotted dimensions.
        step (int): Plot every nth slice along the behavioral axis.
        color_ranges (list[tuple]): List of color-range tuples (start_color, end_color).
        names (list[str]): List of condition names.
        plot_size (int): Size (pixels) of the cubic plot.
        title (str): Title of the plot.
        ba_title (str): Label for behavioral axis in hover info.

    Returns:
        go.Figure: Plotly 3D figure object with interactive hover info.
    """
    fig = go.Figure()

    if names is None:
        names = [f"Condition {i+1}" for i in range(len(manifolds))]

    if color_ranges is None:
        color_ranges = [("black", "gray") for _ in manifolds]

    global_max = 0

    # Determine global symmetric axis limits
    for matrix in manifolds:
        global_max = max(global_max, np.max(np.abs(matrix[:3, :, :])))

    axis_limits = [-global_max, global_max]

    arc_length_axis = np.linspace(0, 1, manifolds[0].shape[1])

    # Plot each manifold separately
    for matrix, behavioral_axis, color_range, name in zip(manifolds, behavioral_axes, color_ranges, names):
        colors = interpolate_colors(color_range[0], color_range[1], len(behavioral_axis))
        selected_indices = list(range(0, len(behavioral_axis), step))

        for idx in selected_indices:
            hover_text = [
                f"<b>{name}</b><br>"
                f"{axis_names[0]}: {x:.2f}<br>"
                f"{axis_names[1]}: {y:.2f}<br>"
                f"{axis_names[2]}: {z:.2f}<br>"
                f"{ba_title}: {behavioral_axis[idx]:.2f}<br>"
                f"Arc-Length: {arc_len:.2f}"
                for x, y, z, arc_len in zip(
                    matrix[0, :, idx], matrix[1, :, idx], matrix[2, :, idx], arc_length_axis
                )
            ]

            fig.add_trace(go.Scatter3d(
                x=matrix[0, :, idx],
                y=matrix[1, :, idx],
                z=matrix[2, :, idx],
                mode='lines',
                line=dict(color=colors[idx], width=4),
                legendgroup=name,
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False
            ))

        # Add invisible trace for selectable legend entry per condition (using second color in tuple)
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines',
            line=dict(color=rgb2hex(to_rgb(f'xkcd:{color_range[1]}')), width=4),
            name=name,
            legendgroup=name,
            showlegend=True
        ))

    # Finalize layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=axis_names[0], range=axis_limits),
            yaxis=dict(title=axis_names[1], range=axis_limits),
            zaxis=dict(title=axis_names[2], range=axis_limits),
            aspectmode='cube'
        ),
        width=plot_size,
        height=plot_size,
        margin=dict(l=0, r=0, b=0, t=50),
        title=title
    )

    return fig


def plot_violin(df, var='RT', group_by='motion_coherence', color_by='choice', colors=None):

    df = df.copy()
    df[group_by] = df[group_by].astype(str)  # Treat group_by as categorical

    if colors is None:
        default_colors = ['xkcd:neon blue', 'xkcd:neon pink', 'xkcd:neon purple',
                          'xkcd:neon yellow', 'xkcd:neon red', 'xkcd:neon green']
        unique_choices = sorted(df[color_by].unique())
        hex_colors = [mcolors.to_hex(mcolors.to_rgb(c)) for c in default_colors[:len(unique_choices)]]
        color_map = dict(zip(unique_choices, hex_colors))
    else:
        unique_choices = sorted(df[color_by].unique())
        color_map = dict(zip(unique_choices, colors[:len(unique_choices)]))

    fig = go.Figure()
    unique_groups = sorted(df[group_by].unique())
    n_choices = len(unique_choices)
    offset = 0.8 / n_choices

    for j, choice in enumerate(unique_choices):
        for i, group in enumerate(unique_groups):
            group_data = df[(df[group_by] == group) & (df[color_by] == choice)]
            fig.add_trace(go.Violin(
                x=[i + (j - (n_choices - 1) / 2) * offset] * len(group_data),
                y=group_data[var],
                name=str(choice),
                legendgroup=str(choice),
                scalegroup=str(choice),
                line_color=color_map[choice],
                box_visible=True,
                meanline_visible=True,
                width=offset
            ))

    fig.update_layout(
        title=f"Distribution of {var} grouped by {group_by} and colored by {color_by}",
        yaxis_title=var,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(unique_groups))),
            ticktext=unique_groups,
            title=group_by
        ),
        template="plotly_white",
        violinmode='overlay',
        width=800,
        height=500
    )

    fig.show()

def plot_retinotopic_2d_hist(df, bin_size=0.1, percentile=10, session='S6'):
    """
    Plot 2D histograms of saccade endpoints for fast and slow RTs in a session.
 
    Parameters:
        df: DataFrame containing EyeX, EyeY, RT, Choice, Session
        bin_size: float, size of the spatial bin (in retinotopic units)
        percentile: float, top/bottom percentile of RT to consider
        session: str, session identifier
    """
    import plotly.graph_objects as go
 
    df = df[df['Session'] == session].dropna(subset=['EyeX', 'EyeY', 'RT'])
    if df.empty:
        raise ValueError(f"No valid data for session {session}.")
 
    centroids = df.attrs.get('centroids', [])
    centroids = {c['choice']: c for c in centroids if c['session'] == session}
    targets = [t for t in df.attrs.get('targets', []) if t['session'] == session]
 
    for tag, rt_filter in zip(['Fast RTs', 'Slow RTs'],
                               [df['RT'] <= df['RT'].quantile(percentile/100),
                                df['RT'] >= df['RT'].quantile(1 - percentile/100)]):
        fig = go.Figure()
        all_x_comb, all_y_comb = [], []
 
        for choice, color in zip([0, 1], ['greys', 'greys']):
            df_sub = df[(df['Choice'] == choice) & rt_filter]
            if df_sub.empty or choice not in centroids:
                continue
 
            cx, cy = centroids[choice]['location_retinotopy']
            x_vals = df_sub['EyeX'].values - cx
            y_vals = df_sub['EyeY'].values - cy
 
            x_min, x_max = x_vals.min(), x_vals.max()
            y_min, y_max = y_vals.min(), y_vals.max()
            x_bins = np.arange(np.floor(x_min / bin_size) * bin_size, np.ceil(x_max / bin_size) * bin_size + bin_size, bin_size)
            y_bins = np.arange(np.floor(y_min / bin_size) * bin_size, np.ceil(y_max / bin_size) * bin_size + bin_size, bin_size)
 
            hist2d, _, _ = np.histogram2d(y_vals, x_vals, bins=(y_bins, x_bins))
            hist2d = np.flipud(hist2d)
 
            x_centers = x_bins[:-1] + bin_size / 2 + cx
            y_centers = y_bins[:-1][::-1] + bin_size / 2 + cy
            fig.add_trace(go.Heatmap(
                z=hist2d,
                x=x_centers,
                y=y_centers,
                colorscale=color,
                colorbar=dict(title=f"Count — Choice {choice}", len=0.45, y=0.75 if choice == 0 else 0.3),
                showscale=True,
                hovertemplate="Choice " + str(choice) + "<br>Count: %{z:.0f}<extra></extra>"
            ))
 
            all_x_comb.append(x_centers)
            all_y_comb.append(y_centers)
 
        # Centroid line
        if len(centroids) == 2:
            xs = [c['location_retinotopy'][0] for c in centroids.values()]
            ys = [c['location_retinotopy'][1] for c in centroids.values()]
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='lines+markers',
                line=dict(dash='dash', color='gray'),
                marker=dict(color='white', size=10, line=dict(color='gray', width=1)),
                showlegend=False,
                hoverinfo='text',
                text=[f"Centroid: ({x:.2f}, {y:.2f})" for x, y in zip(xs, ys)]
            ))
 
        for target in targets:
            if target.get('location_retinotopy') is not None:
                x, y = target['location_retinotopy']
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers',
                    marker=dict(size=10, color='coral', line=dict(color='gray', width=1)),
                    showlegend=False,
                    hoverinfo='text',
                    text=[f"Target: ({x:.2f}, {y:.2f})"]
                ))
 
        if all_x_comb and all_y_comb:
            x_all = np.concatenate(all_x_comb)
            y_all = np.concatenate(all_y_comb)
            x_range = [x_all.min() - bin_size / 2, x_all.max() + bin_size / 2]
            y_range = [y_all.min() - bin_size / 2, y_all.max() + bin_size / 2]
        else:
            x_range, y_range = None, None
 
        fig.update_layout(
            title=f"Retinotopic 2D Histogram — {tag} (Bottom {percentile}%) — Session {session}" if tag == 'Fast RTs' else f"Retinotopic 2D Histogram — {tag} (Top {percentile}%) — Session {session}",
            xaxis_title="Eye X",
            yaxis_title="Eye Y",
            width=750,
            height=650,
            template="plotly_white",
            xaxis=dict(scaleanchor='y', scaleratio=1, range=x_range),
            yaxis=dict(range=y_range),
            margin=dict(l=10, r=10, t=50, b=10),
            hovermode='closest'
        )
        fig.show()

def plot_retinotopic_1d_hist(df, bin_size=0.025, percentile=10, session='S6', use_targets=False):
    """
    Plot 1D histograms of retinotopic EyeX positions (after coordinate transformation) 
    for top and bottom RT percentiles for each choice.
 
    Parameters:
        df: DataFrame from get_saccade_endpoints()
        bin_size: float, size of the histogram bins
        percentile: float, RT percentile threshold
        session: str, session identifier
        use_targets: bool, use targets instead of centroids for transformation
    """
    import plotly.graph_objects as go
    from src.properties import coordinate_transformation
 
    if df.empty:
        raise ValueError("Input DataFrame is empty")
 
    df_trans = coordinate_transformation(df, use_targets=use_targets)
    df_trans = df_trans[df_trans['Session'] == session].copy()
    df_trans = df_trans.dropna(subset=['EyeX', 'EyeY', 'RT'])
 
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=("Choice 0", "Choice 1"))
 
    colors = {'fast': 'steelblue', 'slow': 'goldenrod'}
    for col, choice in enumerate([0, 1], start=1):
        df_choice = df_trans[df_trans['Choice'] == choice]
        if df_choice.empty:
            continue
 
        # RT percentiles
        rt_low = df_choice['RT'].quantile(percentile / 100)
        rt_high = df_choice['RT'].quantile(1 - percentile / 100)
        fast = df_choice[df_choice['RT'] <= rt_low]['EyeX']
        slow = df_choice[df_choice['RT'] >= rt_high]['EyeX']
        if fast.empty and slow.empty:
            continue
 
        # Compute histograms
        x_all = pd.concat([fast, slow])
        bins = np.arange(x_all.min(), x_all.max() + bin_size, bin_size)
 
        fast_hist, _ = np.histogram(fast, bins)
        slow_hist, _ = np.histogram(slow, bins)
        fast_bin_centers = bins[:-1] + bin_size / 2
        slow_bin_centers = bins[:-1] + bin_size / 2
 
        fig.add_trace(go.Bar(
            x=fast_bin_centers,
            y=fast_hist,
            name='Fast RT',
            marker=dict(color=colors['fast'], line=dict(width=1.5, color='black')),
            opacity=0.6,
            showlegend=(col == 1)
        ), row=1, col=col)

        fig.add_trace(go.Bar(
            x=slow_bin_centers,
            y=slow_hist,
            name='Slow RT',
            marker=dict(color=colors['slow'], line=dict(width=1.5, color='black')),
            opacity=0.6,
            showlegend=(col == 1)
        ), row=1, col=col)
 
        # Medians
        fig.add_vline(
            x=fast.median(), line=dict(color=colors['fast'], dash='dot'),
            row=1, col=col
        )
        fig.add_vline(
            x=slow.median(), line=dict(color=colors['slow'], dash='dot'),
            row=1, col=col
        )
 
        # Plot only closest target to the centroid for each choice
        targets = [t for t in df_trans.attrs.get('targets', []) if t['session'] == session]
        centroids = [c for c in df_trans.attrs.get('centroids', []) if c['session'] == session and c['choice'] == choice]
        if targets and centroids:
            centroid_x = centroids[0]['location_retinotopy'][0]
            closest_target = min(
                targets,
                key=lambda t: abs(t['location_retinotopy'][0] - centroid_x)
                if t.get('location_retinotopy') is not None else float('inf')
            )
            loc = closest_target.get('location_retinotopy')
            if loc:
                x = loc[0]
                fig.add_trace(go.Scatter(
                    x=[x], y=[0],
                    mode='markers',
                    marker=dict(size=10, color='coral', line=dict(color='gray', width=1)),
                    showlegend=False
                ), row=1, col=col)
 
        fig.update_xaxes(title_text="Target Line Projection", row=1, col=col)
 
    fig.update_layout(
        title=f"1D Retinotopic Histograms by RT — Session {session} (Bottom/Top {percentile}%)",
        yaxis_title="Count",
        barmode='overlay',
        template="plotly_white",
        width=800,
        height=400,
        margin=dict(t=50, l=40, r=40, b=40)
    )
    fig.show()

def plot_retinotopic_RT_map(df, bin_size=0.1, min_trials=3, session='S6'):
    import plotly.graph_objects as go

    df = df[df['Session'] == session].dropna(subset=['EyeX', 'EyeY', 'RT'])
    if df.empty:
        raise ValueError(f"No valid data for session {session}.")
 
    fig = go.Figure()
    all_x_centers = []
    all_y_centers = []
    all_valid_masks = []
    centroids = df.attrs.get('centroids', [])
    centroids = {c['choice']: c for c in centroids if c['session'] == session}
 
    for choice, cmap in zip([0, 1], ['Viridis', 'Cividis']):
        df_sub = df[df['Choice'] == choice]
        if df_sub.empty or choice not in centroids:
            continue
 
        cx, cy = centroids[choice]['location_retinotopy']
        x_vals = df_sub['EyeX'].values - cx
        y_vals = df_sub['EyeY'].values - cy
        rt_vals = df_sub['RT'].values
 
        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        x_bins = np.arange(np.floor(x_min / bin_size) * bin_size, np.ceil(x_max / bin_size) * bin_size + bin_size, bin_size)
        y_bins = np.arange(np.floor(y_min / bin_size) * bin_size, np.ceil(y_max / bin_size) * bin_size + bin_size, bin_size)
 
        rt_grid = np.full((len(y_bins) - 1, len(x_bins) - 1), np.nan)
        counts = np.zeros_like(rt_grid, dtype=int)
 
        for i in range(len(x_vals)):
            x, y, rt = x_vals[i], y_vals[i], rt_vals[i]
            xi = np.searchsorted(x_bins, x, side='right') - 1
            yi = np.searchsorted(y_bins, y, side='right') - 1
            if 0 <= xi < rt_grid.shape[1] and 0 <= yi < rt_grid.shape[0]:
                if np.isnan(rt_grid[yi, xi]):
                    rt_grid[yi, xi] = 0
                rt_grid[yi, xi] += rt
                counts[yi, xi] += 1
 
        with np.errstate(invalid='ignore'):
            rt_grid = np.where(counts >= min_trials, rt_grid / counts, np.nan)
        rt_grid_masked = np.where(counts >= min_trials, rt_grid, np.nan)
        valid_mask = ~np.isnan(rt_grid)
        all_x_centers.append(x_bins[:-1] + bin_size / 2 + cx)
        all_y_centers.append(y_bins[:-1][::-1] + bin_size / 2 + cy)
        all_valid_masks.append(valid_mask)

        zmin = np.nanmin(rt_grid)
        zmax = np.nanmax(rt_grid)

        colorbar_settings = dict(title=f'Mean RT (ms) — Choice {choice}', len=0.45, y=0.75) if choice==0 else dict(title=f'Mean RT (ms) — Choice {choice}', len=0.45, y=0.3)

        fig.add_trace(go.Heatmap(
            z=np.flipud(rt_grid_masked),
            x=x_bins[:-1] + bin_size / 2 + cx,
            y=y_bins[:-1][::-1] + bin_size / 2 + cy,
            colorscale=cmap,
            colorbar=colorbar_settings,
            zmin=zmin,
            zmax=zmax,
            showscale=True,
            hovertemplate="Choice " + str(choice) + "<br>Mean RT: %{z:.1f} ms<extra></extra>"
        ))


    # Plot centroids
    centroids = df.attrs.get('centroids', [])
    centroids = [c for c in centroids if c['session'] == session]
    centroids = sorted(centroids, key=lambda x: x['choice'])

    if len(centroids) == 2:
        xs = [c['location_retinotopy'][0] for c in centroids]
        ys = [c['location_retinotopy'][1] for c in centroids]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines+markers',
            line=dict(dash='dash', color='gray'),
            marker=dict(color='white', size=10, line=dict(color='gray', width=1)),
            showlegend=False,
            hoverinfo='text',
            text=[f"Centroid: ({x:.2f}, {y:.2f})" for x, y in zip(xs, ys)]
        ))

    # Plot targets
    targets = df.attrs.get('targets', [])
    for target in targets:
        if target['session'] == session and target.get('location_retinotopy') is not None:
            x, y = target['location_retinotopy']
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(size=10, color='coral', line=dict(color='gray', width=1)),
                showlegend=False,
                hoverinfo='text',
                text=[f"Target: ({x:.2f}, {y:.2f})"]
            ))

    if all_valid_masks:
        # Flatten lists of arrays into single arrays based on valid bins
        x_combined = np.concatenate([
            x_center[mask.any(axis=0)] for x_center, mask in zip(all_x_centers, all_valid_masks)
        ])
        y_combined = np.concatenate([
            y_center[mask.any(axis=1)] for y_center, mask in zip(all_y_centers, all_valid_masks)
        ])

        x_range = [x_combined.min() - bin_size / 2, x_combined.max() + bin_size / 2]
        y_range = [y_combined.min() - bin_size / 2, y_combined.max() + bin_size / 2]
    else:
        x_range, y_range = None, None

    # Compute distance between screen targets if available
    target_locs = [t['location_screen'] for t in df.attrs.get('targets', []) if t['session'] == session and t.get('location_screen') is not None]
    if len(target_locs) == 2:
        dist = np.linalg.norm(np.array(target_locs[0]) - np.array(target_locs[1]))
        title_str = f"Retinotopic RT Map — Session {session} (Target Distance: {dist:.1f} dva)"
    else:
        title_str = f"Retinotopic RT Map — Session {session}"
    fig.update_layout(
        title=title_str,
        xaxis_title="Eye X",
        yaxis_title="Eye Y",
        width=750,
        height=650,
        template="plotly_white",
        xaxis=dict(scaleanchor='y', scaleratio=1, range=x_range),
        yaxis=dict(range=y_range),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode='closest'
    )

    fig.show()



def plot_saccade_retinotopy(df, session='S6'):
    """
    Scatter plot of saccade endpoints in retinotopic coordinates, colored by RT.
    Plots choice 0 with Viridis and choice 1 with Magma colormap.
    Centroids are connected with a dashed line. Targets are shown if available.
    """
    import plotly.graph_objects as go

    df = df[df['Session'] == session]
    if df.empty:
        raise ValueError(f"No data found for session {session}")

    fig = go.Figure()

    # Plot Choice 0
    df0 = df[df['Choice'] == 0].dropna(subset=['EyeX', 'EyeY'])
    fig.add_trace(go.Scatter(
        x=df0['EyeX'], y=df0['EyeY'],
        mode='markers',
        marker=dict(
            size=6,
            color=df0['RT'],
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title='RT (ms) — Choice 0', len=0.45, y=0.65)
        ),
        name='Choice 0',
        hoverinfo='text',
        text=[f"RT: {rt:.1f} ms" for rt in df0['RT']]
    ))

    # Plot Choice 1
    df1 = df[df['Choice'] == 1].dropna(subset=['EyeX', 'EyeY'])
    fig.add_trace(go.Scatter(
        x=df1['EyeX'], y=df1['EyeY'],
        mode='markers',
        marker=dict(
            size=6,
            color=df1['RT'],
            colorscale='cividis',
            showscale=True,
            colorbar=dict(title='RT (ms) — Choice 1', len=0.45, y=0.2)
        ),
        name='Choice 1',
        hoverinfo='text',
        text=[f"RT: {rt:.1f} ms" for rt in df1['RT']]
    ))

    # Plot centroids with dashed line
    centroids = df.attrs.get('centroids', [])
    centroids = [c for c in centroids if c['session'] == session]
    centroids = sorted(centroids, key=lambda x: x['choice'])

    if len(centroids) == 2:
        xs = [c['location_retinotopy'][0] for c in centroids]
        ys = [c['location_retinotopy'][1] for c in centroids]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines+markers',
            line=dict(dash='dash', color='gray'),
            marker=dict(color='white', size=10, line=dict(color='gray', width=1)),
            showlegend=False,
            hoverinfo='text',
            text=[f"Centroid: ({x:.2f}, {y:.2f})" for _,(x, y) in enumerate(zip(xs, ys))]
        ))

    # Plot targets
    targets = df.attrs.get('targets', [])
    for target in targets:
        if target['session'] == session and target.get('location_retinotopy') is not None:
            x, y = target['location_retinotopy']
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(size=10, color='coral', line=dict(color='gray', width=1)),
                showlegend=False,
                hoverinfo='text',
                text=[f"Target: ({x:.2f}, {y:.2f})"]
            ))

    # Compute distance between screen targets if available
    target_locs = [t['location_screen'] for t in df.attrs.get('targets', []) if t['session'] == session and t.get('location_screen') is not None]
    if len(target_locs) == 2:
        dist = np.linalg.norm(np.array(target_locs[0]) - np.array(target_locs[1]))
        title_str = f"Saccade Retinotopy — Session {session} (Target Distance: {dist:.1f} dva)"
    else:
        title_str = f"Saccade Retinotopy — Session {session}"
    fig.update_layout(
        title=title_str,
        xaxis_title="Eye X",
        yaxis_title="Eye Y",
        width=750,
        height=650,
        template="plotly_white",
        xaxis=dict(scaleanchor='y', scaleratio=1),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    fig.show()



def plot_visual_field_choice_selectivity_difference(cs_df_session, ds,arc=0.9, percentile=10, cell_range_threshold=0.99, heat=18, mask=None):
    """
    Compute and plot the weighted average response field using CS fast - slow as weights.
 
    Parameters:
        cs_df_session: Single-row DataFrame for a session, containing 'choice_selectivity' and 'cell_range'
        ds: xarray.Dataset with 'ResponseField' and neuron metadata
        arc: float in [0,1], arc slice to select
        percentile: float in (0,50], defines % of RT bins to average at both ends
        cell_range_threshold: float, threshold for filtering weak cells
        heat: float, sharpening factor applied to response field
    """

    assert 0 < percentile <= 50, "Percentile must be in (0, 50]"
    assert 0 <= arc <= 1, "Arc must be in [0, 1]"

    import seaborn as sns
    icefire_colors = sns.color_palette("icefire", as_cmap=False, n_colors=256)
    icefire_colorscale = [[i / 255, mcolors.rgb2hex(c)] for i, c in enumerate(icefire_colors)]

    from src.properties import get_weighted_average_response_field

    # Extract all CS differences (fast - slow)
    row = cs_df_session.iloc[0]
    cs = row['choice_selectivity']  # (neurons, arc, RT)
    cell_range = row['cell_range']
    arc_axis = np.linspace(0, 1, cs.shape[1])
    rt_len = cs.shape[2]
    arc_idx = np.argmin(np.abs(arc_axis - arc))
    bins = max(1, int(rt_len * percentile / 100))

    fast = np.nanmean(cs[:, arc_idx, :bins], axis=1)
    slow = np.nanmean(cs[:, arc_idx, -bins:], axis=1)
    diff = fast - slow
    threshold_mask = cell_range >= cell_range_threshold
    combined_mask = threshold_mask if mask is None else threshold_mask & mask

    valid_indices = np.where(combined_mask)[0]
    print(len(valid_indices), "neurons passed the threshold and mask criteria.")
    if len(valid_indices) == 0:
        raise ValueError("No neurons passed the threshold and mask criteria.")
    all_diffs = diff[valid_indices]

    # Extract and reshape response fields from ds
    Z = ds['ResponseField'].values  # (neurons, x, y)
    Z_flat = Z.reshape(Z.shape[0], -1)

    # Apply the same filtering to Z and compute weighted avg
    Z_masked = Z_flat[valid_indices]
    weighted = get_weighted_average_response_field(all_diffs, Z_masked, heat=heat)
    Z_weighted = weighted.reshape(Z.shape[1:])

    from scipy.ndimage import gaussian_filter
    # Extra smoothing for plotting
    Z_weighted = gaussian_filter(Z_weighted, sigma=3)


    # Plot result
    fig = go.Figure(data=go.Heatmap(
        z=Z_weighted,
        x=ds['x'].values,
        y=ds['y'].values,
        colorscale=icefire_colorscale,
        zmid=0,
        colorbar=dict(title='CS Fast - Slow (Weighted)')
    ))

    fig.update_layout(
        title=f"Weighted Visual Field (CS Fast - Slow) @ Arc {arc:.2f}",
        xaxis=dict(visible=False, showgrid=False, constrain='domain'),
        yaxis=dict(visible=False, showgrid=False, scaleanchor='x', scaleratio=1, constrain='domain'),
        width=500,
        height=500,
        template='plotly_white',
        margin=dict(l=5, r=5, t=30, b=5)
    )
    if 'ChoiceTargets' in ds.attrs:
        for tx, ty in ds.attrs['ChoiceTargets'].values():
            fig.add_trace(go.Scatter(
                x=[tx], y=[ty],
                mode='markers',
                marker=dict(
                    size=12,
                    color='white',
                    line=dict(color='gray', width=1)
                ),
                showlegend=False
            ))
    fig.show()


def plot_proportion(df, choice='Contra', group_by='motion_coherence', targets='target', colors=None):

    df = df.copy()
    df[group_by] = df[group_by].astype(str)

    unique_targets = sorted(df[targets].unique())
    groups = sorted(df[group_by].unique())

    if colors is None:
        default_colors = ['xkcd:neon blue', 'xkcd:neon pink', 'xkcd:neon purple',
                          'xkcd:neon yellow', 'xkcd:neon red', 'xkcd:neon green']
        hex_colors = [mcolors.to_hex(mcolors.to_rgb(c)) for c in default_colors[:len(unique_targets)]]
        color_map = dict(zip(unique_targets, hex_colors))
    else:
        color_map = dict(zip(unique_targets, colors[:len(unique_targets)]))

    fig = go.Figure()

    for target in unique_targets:
        proportions = []
        for g in groups:
            subset = df[(df[group_by] == g) & (df[targets] == target)]
            if len(subset) == 0:
                proportions.append(None)
            else:
                p = (subset['choice'] == choice).mean()
                proportions.append(p)
        fig.add_trace(go.Scatter(
            x=groups,
            y=proportions,
            mode='markers+lines',
            name=str(target),
            marker=dict(color=color_map[target], size=8),
            line=dict(color=color_map[target], width=2, dash='dot')
        ))

    fig.update_layout(
        title=f"Proportion of Trials Choosing '{choice}' by {targets} and {group_by}",
        xaxis_title=group_by,
        yaxis_title=f"Proportion Choosing '{choice}'",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        width=800,
        height=500
    )

    fig.show()



def plot_pop_selectivity_arcslice(df, session='S6', arc=0.75, cell_range_threshold=1.0):
    """
    Plot neuron × behavioral axis matrix for a specific arc-length slice of choice selectivity.

    Parameters:
        df: DataFrame with 'session', 'choice_selectivity', and 'cell_range' columns
        session: str, which session to visualize
        arc: float in [0, 1], position along the arc length to slice (e.g., 0.75)
        cell_range_threshold: float, filter neurons whose min range across choices >= threshold
    """
    # Extract session data
    row = df[df['session'] == session].iloc[0]
    cs = row['choice_selectivity']         # shape: (neurons, arc_length, RT)
    cell_range = row['cell_range']         # shape: (neurons,)
    
    # Filter neurons by dynamic range
    valid_cells = cell_range >= cell_range_threshold
    cs = cs[valid_cells]

    # Determine arc-length index
    arc_len = cs.shape[1]
    arc_axis = np.linspace(0, 1, arc_len)
    arc_idx = np.argmin(np.abs(arc_axis - arc))

    # Extract slice at that arc: shape (neurons, RT)
    cs_slice = cs[:, arc_idx, :]

    # Normalize behavioral axis
    rt_len = cs_slice.shape[1]
    rt_axis = np.linspace(0, 1, rt_len)

    # Sort neurons by (first RT bin - last RT bin) in that slice
    sort_index = np.argsort(np.mean(cs_slice[:, 0:10],axis=-1) - np.mean(cs_slice[:, -10:], axis=1))
    cs_sorted = cs_slice[sort_index]

    # Plot heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cs_sorted,
        x=rt_axis,
        colorscale='earth_r',
        colorbar=dict(title="Choice Selectivity"),
        showscale=True
    ))

    fig.update_layout(
        title=f"Choice Selectivity Raster — Session {session} @ Arc {arc_axis[arc_idx]:.2f}",
        xaxis_title="Normalized Reaction Time",
        yaxis_title="Neuron (sorted)",
        height=500,
        width=650,
        template="plotly_white"
    )

    fig.show()



def plot_pop_selectivity_scatter(cs_df, pops, arc=0.75, percentile=10, cell_range_threshold=1.0,open_interactive=True):
    """
    Interactive scatter plot of population choice selectivity: slow vs fast RT bins.
    
    Displays a scatter plot of neurons' selectivity in slow vs fast reaction time trials,
    colored by log cell range. When hovering over a point, an additional panel shows the
    arc-length responses for that neuron across both choices and RT extremes.
    
    Parameters:
        cs_df: DataFrame with 'session', 'choice_selectivity', 'cell_range'
        pops: DataFrame with 'session', 'choice', 'pop_locav'
        arc: float ∈ [0,1], arc slice to select
        percentile: float ∈ (0,50], defines % of RT bins to average at both ends
        cell_range_threshold: float, threshold for filtering weak cells
    """
    slow_vals, fast_vals, cell_ranges = [], [], []
    contra_slow_all, contra_fast_all = [], []
    ipsi_slow_all, ipsi_fast_all = [], []
    js_labels = []

    for idx, row in cs_df.iterrows():
        cs = row['choice_selectivity']   # shape: (neurons, arc, RT)
        cell_range = row['cell_range']
        session = row['session']

        neurons, arc_len, rt_len = cs.shape
        arc_axis = np.linspace(0, 1, arc_len)
        rt_axis = np.linspace(0, 1, rt_len)

        arc_idx = np.argmin(np.abs(arc_axis - arc))
        bins = max(1, int(rt_len * percentile / 100))

        cs_arc = cs[:, arc_idx, :]
        fast = np.nanmean(cs_arc[:, :bins], axis=1)
        slow = np.nanmean(cs_arc[:, -bins:], axis=1)

        valid = cell_range >= cell_range_threshold

        # Append scatter plot data
        slow_vals.extend(slow[valid])
        fast_vals.extend(fast[valid])
        cell_ranges.extend(cell_range[valid])

        # Compute pop_locav based hover data for this session
        pop_contra = pops[(pops['session'] == session) & (pops['choice'] == 0)].iloc[0]['pop_locav']
        pop_ipsi = pops[(pops['session'] == session) & (pops['choice'] == 1)].iloc[0]['pop_locav']
        rt_len_pop = pop_contra.shape[-1]
        n_pop = max(1, int(rt_len_pop * percentile / 100))
        contra_slow = np.mean(pop_contra[:, :,-n_pop:], axis=2)
        contra_fast = np.mean(pop_contra[:, :,:n_pop], axis=2)
        ipsi_slow = np.mean(pop_ipsi[:, :,-n_pop:], axis=2)
        ipsi_fast = np.mean(pop_ipsi[:, :,:n_pop], axis=2)

        # Append valid neurons' hover data
        contra_slow_all.extend(contra_slow[valid])
        contra_fast_all.extend(contra_fast[valid])
        ipsi_slow_all.extend(ipsi_slow[valid])
        ipsi_fast_all.extend(ipsi_fast[valid])
        js_labels.extend([f"{session} Neuron {i+1}" for i in np.arange(neurons)[valid]])

    # Convert to arrays
    slow_vals = np.array(slow_vals)
    fast_vals = np.array(fast_vals)
    cell_ranges = np.array(cell_ranges)
    log_color = np.log10(cell_ranges )

    # --- PLOT ---
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "histogram"}]],
        shared_xaxes=True,
        shared_yaxes=True,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )

    # Scatter
    fig.add_trace(go.Scatter(
        x=slow_vals,
        y=fast_vals,
        mode='markers',
        marker=dict(
            size=6,
            color=log_color,
            colorscale='Viridis',
            colorbar=dict(
                title='Cell Range (Hz)',
                tickvals=np.concatenate((np.array([0]),np.log10(np.linspace(10,100,10)))),
                ticktext=np.concatenate((np.array([1]),np.linspace(10,100,10))).astype(str)
            )
        ),
        text=[f"{label}<br>Cell Range: {rng:.2f}<br>CS Slow: {x:.2f}<br>CS Fast: {y:.2f}"
              for label, rng, x, y in zip(js_labels, cell_ranges, slow_vals, fast_vals)],
        hoverinfo='text'
    ), row=2, col=1)

    # Histograms
    fig.add_trace(go.Histogram(
        x=slow_vals,
        marker=dict(color='teal'),
        nbinsx=10,
        xbins=dict(start=rt_axis[0], end=rt_axis[-1]),
        showlegend=False,
        opacity=.75
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        y=fast_vals,
        marker=dict(color='teal'),
        nbinsy=10,
        ybins=dict(start=rt_axis[0], end=rt_axis[-1]),
        showlegend=False,
        opacity=.75
    ), row=2, col=2)


    # Layout
    fig.update_layout(
        title=f"Population Choice Selectivity (N={len(cell_ranges)}, arc = {arc:.2f}, top/bottom {percentile}% RT)",
        xaxis3=dict(title="Choice Selectivity (Slow RT)", scaleanchor="y3", scaleratio=1),
        yaxis3=dict(title="Choice Selectivity (Fast RT)"),
        width=700,
        height=700,
        template="plotly_white",
        autosize=False,
        showlegend=False
    )
    fig.show()

    # Generate interactive figure to explore single neuron activity 
    import json
    contra_slow_json = json.dumps(np.array(contra_slow_all).tolist())
    contra_fast_json = json.dumps(np.array(contra_fast_all).tolist())
    ipsi_slow_json = json.dumps(np.array(ipsi_slow_all).tolist())
    ipsi_fast_json = json.dumps(np.array(ipsi_fast_all).tolist())
    arc_axis_json = json.dumps(arc_axis.tolist())
    labels_json = json.dumps(js_labels)

    scatter_html = fig.to_html(include_plotlyjs='cdn', full_html=False, div_id='scatter-plot')

    # JavaScript to update the second figure on hover
    hover_js = f"""
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        const scatter = document.getElementById('scatter-plot');
        scatter.on('plotly_hover', function(event) {{
            const idx = event.points[0].pointIndex;
            const arcAxis = {arc_axis_json};
            const traces = [
                {{
                    x: arcAxis,
                    y: {contra_slow_json}[idx],
                    mode: 'lines',
                    name: 'Contra Slow',
                    line: {{color: 'a8ce84'}}
                }},
                {{
                    x: arcAxis,
                    y: {ipsi_slow_json}[idx],
                    mode: 'lines',
                    name: 'Ipsi Slow',
                    line: {{color: 'f9b01b'}}
                }},
                {{
                    x: arcAxis,
                    y: {contra_fast_json}[idx],
                    mode: 'lines',
                    name: 'Contra Fast',
                    line: {{color: '1272a0'}}
                }},
                {{
                    x: arcAxis,
                    y: {ipsi_fast_json}[idx],
                    mode: 'lines',
                    name: 'Ipsi Fast',
                    line: {{color: 'e8355b'}}
                }}
            ];
            const layout = {{
                title: {labels_json}[idx],
                xaxis: {{title: 'Arc-Length'}},
                yaxis: {{title: 'Response (Hz)'}}
            }};
            Plotly.newPlot('hover-plot', traces, layout);
        }});
    }});
    </script>
    """

    # Combined layout with scatter on the left and hover plot on the right
    html_combined = f'''
    <div style="display: flex; justify-content: space-between;">
        <div style="flex: 1; min-width: 700px;" id="scatter-container">
            {scatter_html}
        </div>
        <div style="flex: 1; min-width: 600px;" id="hover-plot"></div>
    </div>
    {hover_js}
    '''


    # Create figures directory if it doesn't exist
    import os
    os.makedirs("figures", exist_ok=True)
    # Save the full HTML to disk
    filename = "figures/choice_selectivity_scatter.html"
    with open(filename, 'w') as f:
        f.write(html_combined)

    # Always open in the system's default browser
    if open_interactive:
        import webbrowser, os
        webbrowser.open_new_tab('file://' + os.path.realpath(filename))

    # Histogram of CS difference (Fast - Slow)
    diff_mask = (fast_vals > 0.5) | (slow_vals > 0.5)
    cs_diff = fast_vals[diff_mask] - slow_vals[diff_mask]

    fig_diff = go.Figure()
    fig_diff.add_trace(go.Histogram(
        x=cs_diff,
        xbins=dict(start=-1,end=1,size=2/21),
        marker=dict(color='steelblue'),
        opacity=0.75
    ))
    fig_diff.update_layout(
        title=f"CS Fast - CS Slow (Only where either CS > 0.5, N={len(cs_diff)})",
        xaxis_title="CS Fast - CS Slow",
        yaxis_title="Count",
        template="plotly_white",
        width=700,
        height=400
    )
    fig_diff.show()


import xarray as xr
import plotly.subplots as sp
import plotly.graph_objects as go

def plot_response_fields(ds, cells='TinC', n_columns=10, sort=None):
    """
    Plot smoothed response fields for a selection of neurons from an xarray Dataset.

    Parameters:
        ds: xarray.Dataset returned from get_response_fields
        cells: list of integers (cell indices) or string key in ds.attrs ('TinC', 'TinI', etc.)
        n_columns: int, number of subplot columns
    """
    if isinstance(cells, str):
        cells = ds.attrs.get(cells, [])
    if not cells:
        raise ValueError("No valid cells specified to plot.")
    
    if sort is not None and sort in ds.attrs:
        sort_array = np.array(ds.attrs[sort])
        cells = list(np.array(cells)[np.argsort(sort_array[cells])[::-1]])
    
    n_cells = len(cells)
    n_rows = (n_cells + n_columns - 1) // n_columns
    choice_targets = ds.attrs.get('ChoiceTargets', {})
    target_coords = list(choice_targets.values())

    fig = sp.make_subplots(
        rows=n_rows, cols=n_columns,
        horizontal_spacing=0.005, vertical_spacing=0.01
    )

    for idx, cell in enumerate(cells):
        row = idx // n_columns + 1
        col = idx % n_columns + 1
        z = ds['ResponseField'].sel(cell=cell)
        heatmap = go.Heatmap(
            z=z.values,
            x=ds['x'].values,
            y=ds['y'].values,
            colorscale='Inferno',
            zmin=0,
            showscale=False
        )
        fig.add_trace(heatmap, row=row, col=col)
        for tx, ty in target_coords:
            fig.add_trace(go.Scatter(
                x=[tx], y=[ty],
                mode='markers',
                marker=dict(
                    size=6,
                    color='white',
                    line=dict(color='gray', width=1)
                ),
                showlegend=False
            ), row=row, col=col)
        fig.update_xaxes(
            visible=False,
            showgrid=False,
            constrain='domain',
            row=row, col=col
        )
        fig.update_yaxes(
            visible=False,
            showgrid=False,
            scaleanchor=f"x{(row - 1) * n_columns + col}",
            scaleratio=1,
            constrain='domain',
            row=row, col=col
        )
        
        max_fr = np.nanmax(z.values)
        fig.add_annotation(
            text=f"{max_fr:.1f} Hz",
            xref=f"x{(row-1)*n_columns+col}",
            yref=f"y{(row-1)*n_columns+col}",
            x=ds['x'].values[-1],
            y=ds['y'].values[-1],
            showarrow=False,
            font=dict(size=10, color="white"),
            xanchor='right',
            yanchor='top',
            row=row, col=col
        )
        fig.add_annotation(
            text=f"Cell {cell}",
            xref=f"x{(row-1)*n_columns+col}",
            yref=f"y{(row-1)*n_columns+col}",
            x=ds['x'].values[0],
            y=ds['y'].values[-1],
            showarrow=False,
            font=dict(size=10, color="white"),
            xanchor='left',
            yanchor='top',
            row=row, col=col
        )
        morans_i = ds.attrs.get('MoransI', [np.nan] * len(ds['cell']))[cell]
        sic = ds.attrs.get('SpatialInformation', [np.nan] * len(ds['cell']))[cell]

        # Moran's I in bottom left
        fig.add_annotation(
            text=f"MI: {morans_i:.2f}",
            xref=f"x{(row-1)*n_columns+col}",
            yref=f"y{(row-1)*n_columns+col}",
            x=ds['x'].values[0],
            y=ds['y'].values[0],
            showarrow=False,
            font=dict(size=10, color="white"),
            xanchor='left',
            yanchor='bottom',
            row=row, col=col
        )

        # SIC in bottom right
        fig.add_annotation(
            text=f"SI: {sic:.2f}",
            xref=f"x{(row-1)*n_columns+col}",
            yref=f"y{(row-1)*n_columns+col}",
            x=ds['x'].values[-1],
            y=ds['y'].values[0],
            showarrow=False,
            font=dict(size=10, color="white"),
            xanchor='right',
            yanchor='bottom',
            row=row, col=col
        )

    cell_size = 130  # pixels per subplot
    fig.update_layout(
        height=cell_size * n_rows,
        width=cell_size * n_columns,
        showlegend=False,
        title_text="Population Response Fields",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.show()



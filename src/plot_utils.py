import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap, to_rgb, rgb2hex
from scipy.ndimage import gaussian_filter1d as gf

from plotly.subplots import make_subplots

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




def plot_local_average(data, behavioral_axis, arc_length_axis=None, top_colors=("black", "light gray"),
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

    smoothed_data = gf(data, sigma=smooth_sigma, axis=0)
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

"""
Plotting utilities for GAOT results visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Tuple
import matplotlib.colors as mcolors

import matplotlib

########################################################
# Plotting settings
########################################################
# Dark colors
C_BLACK = '#000000'
C_WHITE = '#ffffff'
C_BLUE = '#093691'
C_RED = '#911b09'
C_BLACK_BLUEISH = '#011745'
C_BLACK_REDDISH = '#380801'
C_WHITE_BLUEISH = '#dce5f5'
C_WHITE_REDDISH = '#f5dcdc'
# Bright colors
C_BRIGHT_PURPLE = '#7f00ff'   
C_BRIGHT_PINK   = '#ff00ff'   
C_BRIGHT_ORANGE = '#ff7700'   
C_BRIGHT_YELLOW = '#ffdd00'   
C_BRIGHT_GREEN  = '#00ee00'   
C_BRIGHT_CYAN   = '#00ffff'   
C_BRIGHT_BLUE   = '#0f00ff'   
CMAP_BWR = matplotlib.colors.LinearSegmentedColormap.from_list(
  'blue_white_red',
  [C_BLACK_BLUEISH, C_BLUE, C_WHITE, C_RED, C_BLACK_REDDISH],
  N=200,
)
CMAP_WRB = matplotlib.colors.LinearSegmentedColormap.from_list(
  'white_red_black',
  [C_WHITE, C_RED, C_BLACK],
  N=200,
)
# Scatter settings
SCATTER_SETTINGS = dict(marker='s', s=1, alpha=1, linewidth=0)
HATCH_SETTINGS = dict(facecolor='#b8b8b8', edgecolor='#4f4f4f', linewidth=.0)

########################################################
# Plotting functions
########################################################
def plot_estimates(
    u_inp: np.ndarray,
    u_gtr: np.ndarray,
    u_prd: np.ndarray,
    x_inp: np.ndarray,
    x_out: np.ndarray,
    symmetric: Union[bool, List[bool]] = True,
    names: Optional[List[str]] = None,
    domain: Tuple[List[float], List[float]] = ([-1, -1], [1, 1]),
    colorbar_type: str = "light",
    show_error: bool = True
) -> plt.Figure:
    """
    Plots input data, ground-truth, model predictions, and optionally absolute errors over a 2D domain.

    This function creates a figure with three or four panels (columns) for each variable:
    1) Input data,
    2) Ground-truth values,
    3) Model predictions,
    4) Absolute error (|ground-truth - prediction|) - optional based on show_error parameter.

    A horizontal colorbar is provided for each column, showing the data range used for coloring.
    
    Parameters
    ----------
    u_inp : np.ndarray
        The input data array of shape (N_inp, n_input_vars), where:
          - N_inp is the number of input points.
          - n_input_vars is the number of input variables (e.g., different physical quantities).
    u_gtr : np.ndarray
        The ground-truth data array of shape (N_out, n_output_vars). N_out can differ from N_inp
        if the input and output grids do not match.
    u_prd : np.ndarray
        The model-predicted data array, same shape as `u_gtr` (i.e., (N_out, n_output_vars)).
        This is compared against `u_gtr` to compute the absolute error.
    x_inp : np.ndarray
        The (x, y) coordinates of each input point, shape (N_inp, 2).
        Used for the scatter plot of `u_inp`.
    x_out : np.ndarray
        The (x, y) coordinates for the output/ground-truth grid, shape (N_out, 2).
        Used for the scatter plots of `u_gtr`, `u_prd`, and their absolute error.
    symmetric : bool or list of bool, optional
        Whether to use a symmetric color scale (colormap) for each variable. 
        If True, the color limits are set to [-vmax, +vmax], where vmax is 
        the maximum absolute value across data samples for that variable. 
        If a list of booleans is provided, each element corresponds to one variable.
    names : list of str, optional
        A list of variable names (of length n_vars) used as labels on the vertical axis.
        If None, default labels such as "Variable 00", "Variable 01", etc., are used.
    domain : tuple of list, optional
        Defines the displayed plotting region as ([x_min, y_min], [x_max, y_max]).
        Defaults to ([0, 0], [1, 1]). The background hatch pattern will fill this region.
    colorbar_type: str, optional
        The type of colorbar to use. Can be "light" or "dark". Defaults to "light".
    show_error: bool, optional
        Whether to show the absolute error column. Defaults to True.
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the subplots. Each variable has one row in the figure,
        and there are three or four columns of scatter plots: input, ground-truth, prediction,
        and optionally absolute error. Each column is accompanied by a horizontal colorbar.

    Notes
    -----
    - Internally, the function arranges subplots for each variable in a two-row layout:
      the top row is for the actual scatter plots, and the bottom row hosts the colorbars.
    - The absolute error is plotted as |u_gtr - u_prd|.
    - Use the returned figure object to further customize, save, or display the figure.

    Examples
    --------
    >>> import numpy as np
    >>> # Assume we have two variables (n_vars = 2)
    >>> # Input grid has 50 points, output grid has 100 points
    >>> x_inp = np.random.rand(50, 2)
    >>> x_out = np.random.rand(100, 2)
    >>> u_inp = np.random.randn(50, 2)
    >>> u_gtr = np.random.randn(100, 2)
    >>> u_prd = u_gtr + 0.1 * np.random.randn(100, 2)
    >>> fig = plot_estimates(
    ...     u_inp=u_inp,
    ...     u_gtr=u_gtr,
    ...     u_prd=u_prd,
    ...     x_inp=x_inp,
    ...     x_out=x_out,
    ...     symmetric=True,
    ...     names=["Temperature", "Concentration"],
    ...     domain=([0, 0], [1, 1])
    ... )
    >>> fig.tight_layout()
    >>> fig.show()
    """
    _HEIGHT_PER_ROW = 1.9
    _HEIGHT_MARGIN = .2
    _SCATTER_SETTINGS = SCATTER_SETTINGS.copy()
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * .4 * _HEIGHT_PER_ROW
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * 128 / (x_inp.shape[0] ** .5)

    n_vars = u_gtr.shape[-1]
    if isinstance(symmetric, bool):
        symmetric = [symmetric] * n_vars

    # Calculate number of columns and adjust figsize accordingly
    n_cols = 4 if show_error else 3
    base_width = 8.6  # Original width for 4 columns
    figsize = (base_width * n_cols / 4.0, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN)
    fig = plt.figure(figsize=figsize)
    g_fig = fig.add_gridspec(
        nrows=n_vars,
        ncols=1,
        wspace=0,
        hspace=0,
    )

    figs = []
    for ivar in range(n_vars):
        figs.append(fig.add_subfigure(g_fig[ivar], frameon=False))
    # Add axes
    axs_inp = []
    axs_gtr = []
    axs_prd = []
    axs_err = []
    axs_cb_inp = []
    axs_cb_out = []
    axs_cb_err = []
    for ivar in range(n_vars):
        g = figs[ivar].add_gridspec(
        nrows=2,
        ncols=n_cols,
        height_ratios=[1, .05],
        wspace=0.20,
        hspace=0.05,
        )
        axs_inp.append(figs[ivar].add_subplot(g[0, 0]))
        axs_gtr.append(figs[ivar].add_subplot(g[0, 1]))
        axs_prd.append(figs[ivar].add_subplot(g[0, 2]))
        if show_error:
            axs_err.append(figs[ivar].add_subplot(g[0, 3]))
        else:
            axs_err.append(None)  # Placeholder to maintain indexing
        
        axs_cb_inp.append(figs[ivar].add_subplot(g[1, 0]))
        if show_error:
            axs_cb_out.append(figs[ivar].add_subplot(g[1, 1:3]))
            axs_cb_err.append(figs[ivar].add_subplot(g[1, 3]))
        else:
            axs_cb_out.append(figs[ivar].add_subplot(g[1, 1:3]))  # Spans to the end
            axs_cb_err.append(None)  # Placeholder
    # Settings
    all_axes = [axs_inp, axs_gtr, axs_prd]
    if show_error:
        all_axes.append(axs_err)
    for ax in [ax for axs in all_axes for ax in axs if ax is not None]:
        ax: plt.Axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([domain[0][0], domain[1][0]])
        ax.set_ylim([domain[0][1], domain[1][1]])
        ax.fill_between(
        x=[domain[0][0], domain[1][0]], y1=domain[0][1], y2=domain[1][1],
        **HATCH_SETTINGS,
        )

    # Get prediction error
    u_err = (u_gtr - u_prd)

    # Choose colormap based on colorbar_type
    if colorbar_type == "light":
        cmap_symmetric = plt.cm.jet
        cmap_asymmetric = plt.cm.jet
    else:
        cmap_symmetric = CMAP_BWR
        cmap_asymmetric = CMAP_WRB

    # Loop over variables
    for ivar in range(n_vars):
        # Get ranges
        vmax_inp = np.max(u_inp[:, ivar])
        vmax_gtr = np.max(u_gtr[:, ivar])
        vmax_prd = np.max(u_prd[:, ivar])
        vmax_out = max(vmax_gtr, vmax_prd)
        vmin_inp = np.min(u_inp[:, ivar])
        vmin_gtr = np.min(u_gtr[:, ivar])
        vmin_prd = np.min(u_prd[:, ivar])
        vmin_out = min(vmin_gtr, vmin_prd)
        abs_vmax_inp = max(np.abs(vmax_inp), np.abs(vmin_inp))
        abs_vmax_out = max(np.abs(vmax_out), np.abs(vmin_out))

        # Plot input
        h = axs_inp[ivar].scatter(
        x=x_inp[:, 0],
        y=x_inp[:, 1],
        c=u_inp[:, ivar],
        cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
        vmax=(abs_vmax_inp if symmetric[ivar] else vmax_inp),
        vmin=(-abs_vmax_inp if symmetric[ivar] else vmin_inp),
        **_SCATTER_SETTINGS,
        )
        cb = plt.colorbar(h, cax=axs_cb_inp[ivar], orientation='horizontal')
        cb.formatter.set_powerlimits((-0, 0))
        # Plot ground truth
        h = axs_gtr[ivar].scatter(
        x=x_out[:, 0],
        y=x_out[:, 1],
        c=u_gtr[:, ivar],
        cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
        vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
        vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
        **_SCATTER_SETTINGS,
        )
        # Plot estimate
        h = axs_prd[ivar].scatter(
        x=x_out[:, 0],
        y=x_out[:, 1],
        c=u_prd[:, ivar],
        cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
        vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
        vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
        **_SCATTER_SETTINGS,
        )
        cb = plt.colorbar(h, cax=axs_cb_out[ivar], orientation='horizontal')
        cb.formatter.set_powerlimits((-0, 0))

        # Plot error (only if show_error is True)
        if show_error:
            h = axs_err[ivar].scatter(
            x=x_out[:, 0],
            y=x_out[:, 1],
            c=np.abs(u_err[:, ivar]),
            cmap=cmap_asymmetric,
            vmin=0,
            vmax=np.max(np.abs(u_err[:, ivar])),
            **_SCATTER_SETTINGS,
            )
            cb = plt.colorbar(h, cax=axs_cb_err[ivar], orientation='horizontal')
            cb.formatter.set_powerlimits((-0, 0))

    # Set titles
    axs_inp[0].set(title='Input')
    axs_gtr[0].set(title='Ground-truth')
    axs_prd[0].set(title='Model estimate')
    if show_error:
        axs_err[0].set(title='Absolute error')

    # Set variable names
    for ivar in range(n_vars):
        label = names[ivar] if names else f'Variable {ivar:02d}'
        axs_inp[ivar].set(ylabel=label);

    # Rotate colorbar tick labels
    cb_axes = [axs_cb_inp, axs_cb_out]
    if show_error:
        cb_axes.append(axs_cb_err)
    for ax in [ax for axs in cb_axes for ax in axs if ax is not None]:
        ax: plt.Axes
        ax.xaxis.get_offset_text().set(size=8)
        ax.xaxis.set_tick_params(labelsize=8)

    return fig


def create_sequential_animation(gt_sequence: np.ndarray, pred_sequence: np.ndarray,
                               coords: np.ndarray, save_path: str,
                               input_data: np.ndarray = None,
                               time_values: List[float] = None,
                               interval: int = 500, symmetric: Union[bool, List[bool]] = True,
                               domain: Tuple[List[float], List[float]] = None,
                               names: List[str] = None,
                               colorbar_type: str = "light",
                               show_error: bool = True,
                               dynamic_colorscale: bool = True,
                               u_mean: np.ndarray = None,
                               u_std: np.ndarray = None) -> None:
    """
    Create animation comparing input, ground truth and prediction sequences.
    Uses 3 or 4-column layout identical to plot_estimates: Input | Ground Truth | Prediction | [Error]
    
    Args:
        gt_sequence: Ground truth sequence [n_timesteps, n_points, n_channels] (denormalized)
        pred_sequence: Prediction sequence [n_timesteps, n_points, n_channels] (denormalized)
        coords: Coordinates [n_points, coord_dim]
        save_path: Path to save animation
        input_data: Input data [n_points, n_channels] (static, shown in first column)
        time_values: List of time values for each frame
        interval: Interval between frames in milliseconds
        symmetric: Whether to use symmetric colorscale (bool or list of bool)
        domain: Plotting domain ([x_min, y_min], [x_max, y_max])
        names: Variable names for each channel
        colorbar_type: The type of colorbar to use ("light" or "dark")
        show_error: Whether to show the relative error column (defaults to True)
        dynamic_colorscale: Whether to use per-frame dynamic colorscales (default True)
        u_mean: Normalization mean [1, n_channels] for error computation
        u_std: Normalization std [1, n_channels] for error computation
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Matplotlib animation not available")
        return
    
    if coords.shape[1] != 2:
        print("Animation currently only supports 2D coordinates")
        return
    
    n_timesteps, n_points, n_channels = gt_sequence.shape
    
    _HEIGHT_PER_ROW = 1.9
    _HEIGHT_MARGIN = .2
    _SCATTER_SETTINGS = SCATTER_SETTINGS.copy()
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * .4 * _HEIGHT_PER_ROW
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * 128 / (coords.shape[0] ** .5)
    
    if isinstance(symmetric, bool):
        symmetric = [symmetric] * n_channels
    
    if colorbar_type == "light":
        cmap_symmetric = plt.cm.jet
        cmap_asymmetric = plt.cm.jet
    else:
        cmap_symmetric = CMAP_BWR
        cmap_asymmetric = CMAP_WRB
    
    # Calculate number of columns and adjust figsize accordingly
    n_cols = 4 if show_error else 3
    base_width = 8.6  # Original width for 4 columns
    # Make figure much taller for progress bar well below at bottom
    figsize = (base_width * n_cols / 4.0, _HEIGHT_PER_ROW*n_channels+_HEIGHT_MARGIN+6.0)
    fig = plt.figure(figsize=figsize)
    g_fig = fig.add_gridspec(
        nrows=n_channels,
        ncols=1,
        wspace=0,
        hspace=0,
    )

    figs = []
    for ivar in range(n_channels):
        figs.append(fig.add_subfigure(g_fig[ivar], frameon=False))
    
    if domain is not None:
        plot_domain = domain
    else:
        plot_domain = ([coords[:, 0].min(), coords[:, 1].min()], 
                      [coords[:, 0].max(), coords[:, 1].max()])
    
    scatter_objects = {'inp': [], 'gt': [], 'pred': [], 'error': []}
    axes_inp = []
    axes_gt = []
    axes_pred = []
    axes_err = []
    axes_cb_inp = []
    axes_cb_gt = []
    axes_cb_err = []
    
    # Add axes for each channel following plot_estimates pattern exactly
    for ivar in range(n_channels):
        g = figs[ivar].add_gridspec(
            nrows=2,
            ncols=n_cols,
            height_ratios=[1, .05],
            wspace=0.20,
            hspace=0.05,
        )
        axes_inp.append(figs[ivar].add_subplot(g[0, 0]))
        axes_gt.append(figs[ivar].add_subplot(g[0, 1]))
        axes_pred.append(figs[ivar].add_subplot(g[0, 2]))
        if show_error:
            axes_err.append(figs[ivar].add_subplot(g[0, 3]))
        else:
            axes_err.append(None)  # Placeholder
            
        axes_cb_inp.append(figs[ivar].add_subplot(g[1, 0]))
        if show_error:
            axes_cb_gt.append(figs[ivar].add_subplot(g[1, 1:3]))  # Spans 2 columns like plot_estimates
            axes_cb_err.append(figs[ivar].add_subplot(g[1, 3]))
        else:
            axes_cb_gt.append(figs[ivar].add_subplot(g[1, 1:3]))  # Spans to the end
            axes_cb_err.append(None)  # Placeholder
    
    # Settings for all axes (same as plot_estimates)
    all_axes = [axes_inp, axes_gt, axes_pred]
    if show_error:
        all_axes.append(axes_err)
    for ax in [ax for axs in all_axes for ax in axs if ax is not None]:
        ax: plt.Axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([plot_domain[0][0], plot_domain[1][0]])
        ax.set_ylim([plot_domain[0][1], plot_domain[1][1]])
        ax.fill_between(
            x=[plot_domain[0][0], plot_domain[1][0]], 
            y1=plot_domain[0][1], y2=plot_domain[1][1],
            **HATCH_SETTINGS,
        )
    
    u_err_0 = (gt_sequence[0] - pred_sequence[0])
    
    for ivar in range(n_channels):
        gt_all = gt_sequence[:, :, ivar]
        pred_all = pred_sequence[:, :, ivar]
        
        vmax_gtr = np.max(gt_all)
        vmax_prd = np.max(pred_all)
        vmax_out = max(vmax_gtr, vmax_prd)
        vmin_gtr = np.min(gt_all)
        vmin_prd = np.min(pred_all)
        vmin_out = min(vmin_gtr, vmin_prd)
        abs_vmax_out = max(np.abs(vmax_out), np.abs(vmin_out))
        
        if input_data is not None:
            vmax_inp = np.max(input_data[:, ivar])
            vmin_inp = np.min(input_data[:, ivar])
            abs_vmax_inp = max(np.abs(vmax_inp), np.abs(vmin_inp))
            
            h_inp = axes_inp[ivar].scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                c=input_data[:, ivar],
                cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
                vmax=(abs_vmax_inp if symmetric[ivar] else vmax_inp),
                vmin=(-abs_vmax_inp if symmetric[ivar] else vmin_inp),
                **_SCATTER_SETTINGS,
            )
            scatter_objects['inp'].append(h_inp)
            cb_inp = plt.colorbar(h_inp, cax=axes_cb_inp[ivar], orientation='horizontal')
            cb_inp.formatter.set_powerlimits((-0, 0))
        else:
            h_inp = axes_inp[ivar].scatter([], [], **_SCATTER_SETTINGS)
            scatter_objects['inp'].append(h_inp)
        
        # Plot ground truth
        h_gt = axes_gt[ivar].scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            c=gt_sequence[0, :, ivar],
            cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
            vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
            vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
            **_SCATTER_SETTINGS,
        )
        scatter_objects['gt'].append(h_gt)
        cb_gt = plt.colorbar(h_gt, cax=axes_cb_gt[ivar], orientation='horizontal')
        cb_gt.formatter.set_powerlimits((-0, 0))
        
        # Plot prediction
        h_pred = axes_pred[ivar].scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            c=pred_sequence[0, :, ivar],
            cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
            vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
            vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
            **_SCATTER_SETTINGS,
        )
        scatter_objects['pred'].append(h_pred)
        cb_pred = plt.colorbar(h_pred, cax=axes_cb_gt[ivar], orientation='horizontal')
        cb_pred.formatter.set_powerlimits((-0, 0))
        
        if show_error:
            # Compute relative error in NORMALIZED space (like compute_batch_errors)
            # For spatial visualization, we show pointwise errors
            # But note: the printed mean uses sum-based formula for consistency
            
            if u_mean is not None and u_std is not None:
                # Normalize both sequences for this channel
                mean_ch = u_mean[0, ivar] if u_mean.ndim > 1 else u_mean[ivar]
                std_ch = u_std[0, ivar] if u_std.ndim > 1 else u_std[ivar]
                
                gt_norm_0 = (gt_sequence[0, :, ivar] - mean_ch) / std_ch
                pred_norm_0 = (pred_sequence[0, :, ivar] - mean_ch) / std_ch
                
                # Pointwise relative error in normalized space for spatial visualization
                abs_err_norm_0 = np.abs(pred_norm_0 - gt_norm_0)
                gt_abs_norm_0 = np.abs(gt_norm_0)
                rel_err_0 = (abs_err_norm_0 / (gt_abs_norm_0 + 1e-10)) * 100.0
                
                # Find max relative error across all frames (for colorbar range)
                all_rel_errors = []
                for fr in range(gt_sequence.shape[0]):
                    gt_norm_fr = (gt_sequence[fr, :, ivar] - mean_ch) / std_ch
                    pred_norm_fr = (pred_sequence[fr, :, ivar] - mean_ch) / std_ch
                    abs_err_norm_fr = np.abs(pred_norm_fr - gt_norm_fr)
                    gt_abs_norm_fr = np.abs(gt_norm_fr)
                    rel_err_fr = (abs_err_norm_fr / (gt_abs_norm_fr + 1e-10)) * 100.0
                    all_rel_errors.append(rel_err_fr.max())
                max_rel_err = max(all_rel_errors)
            else:
                # Fallback: pointwise error in physical space (not recommended)
                numerator_0 = np.abs(pred_sequence[0, :, ivar] - gt_sequence[0, :, ivar])
                denominator_0 = np.abs(gt_sequence[0, :, ivar]) + 1e-10
                rel_err_0 = (numerator_0 / denominator_0) * 100.0
                
                all_rel_errors = []
                for fr in range(gt_sequence.shape[0]):
                    numerator_fr = np.abs(pred_sequence[fr, :, ivar] - gt_sequence[fr, :, ivar])
                    denominator_fr = np.abs(gt_sequence[fr, :, ivar]) + 1e-10
                    rel_err_fr = (numerator_fr / denominator_fr) * 100.0
                    all_rel_errors.append(rel_err_fr.max())
                max_rel_err = max(all_rel_errors)
            
            # Fixed colorscale from 1% to 100% for error plots
            # Clamp rel_err_0 to 100% max
            rel_err_0_clamped = np.clip(rel_err_0, 1.0, 100.0)
            
            h_err = axes_err[ivar].scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                c=rel_err_0_clamped,
                cmap=cmap_asymmetric,
                vmin=1.0,
                vmax=100.0,
                **_SCATTER_SETTINGS,
            )
            scatter_objects['error'].append(h_err)
            cb_err = plt.colorbar(h_err, cax=axes_cb_err[ivar], orientation='horizontal')
            cb_err.set_label('Relative Error (%, clamped 1-100)', fontsize=8)
        else:
            scatter_objects['error'].append(None)  # Placeholder
    
    axes_inp[0].set(title='Input')
    axes_gt[0].set(title='Ground truth')
    axes_pred[0].set(title='Prediction')
    if show_error:
        axes_err[0].set(title='Relative Error (%)')
    
    for ivar in range(n_channels):
        label = names[ivar] if names and ivar < len(names) else f'Variable {ivar:02d}'
        axes_inp[ivar].set(ylabel=label)
    
    cb_axes = [axes_cb_inp, axes_cb_gt]
    if show_error:
        cb_axes.append(axes_cb_err)
    for ax in [ax for axs in cb_axes for ax in axs if ax is not None]:
        ax: plt.Axes
        ax.xaxis.get_offset_text().set(size=8)
        ax.xaxis.set_tick_params(labelsize=8)
    
    # Adjust figure to make room for progress bar way below at bottom
    fig.subplots_adjust(bottom=0.35)  # Even more space at bottom
    
    # Add progress bar to figure (way below all subplots and axis labels)
    # Position: [left, bottom, width, height] in figure coordinates
    progress_ax = fig.add_axes([0.15, 0.02, 0.7, 0.012])  # Even lower in extended figure
    progress_bar = progress_ax.barh([0], [0], height=0.8, color='steelblue', alpha=0.7)
    progress_ax.set_xlim(0, n_timesteps)
    progress_ax.set_ylim(-0.5, 0.5)
    progress_ax.axis('off')
    
    # Add time label with total duration
    # Compute total duration if time_values provided
    total_duration_ns = None
    if time_values and len(time_values) > 0:
        # Assume uniform spacing or use actual time differences
        total_duration_ns = (time_values[-1] - time_values[0]) * 1e9 if time_values[-1] < 1e-6 else time_values[-1]
        if total_duration_ns < 1e-3:  # If in seconds, convert to ns
            total_duration_ns *= 1e9
    
    progress_text = progress_ax.text(0.5, -3.0, '', ha='center', va='top', 
                                     transform=progress_ax.transAxes, fontsize=10, 
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def animate(frame):
        """Update function for animation."""
        # Compute relative error matching compute_batch_errors EXACTLY
        # Sum errors across ALL channels, then divide by total (not per-channel average!)
        if show_error:
            if u_mean is not None and u_std is not None:
                # Sum errors across ALL channels and space (matching compute_batch_errors)
                total_error_sum = 0.0
                total_gt_sum = 0.0
                
                for ivar in range(n_channels):
                    gt_frame = gt_sequence[frame, :, ivar]
                    pred_frame = pred_sequence[frame, :, ivar]
                    
                    # Normalize
                    mean_ch = u_mean[0, ivar] if u_mean.ndim > 1 else u_mean[ivar]
                    std_ch = u_std[0, ivar] if u_std.ndim > 1 else u_std[ivar]
                    gt_norm = (gt_frame - mean_ch) / std_ch
                    pred_norm = (pred_frame - mean_ch) / std_ch
                    
                    # Accumulate sums across all channels
                    total_error_sum += np.abs(pred_norm - gt_norm).sum()
                    total_gt_sum += np.abs(gt_norm).sum()
                
                # Compute relative error across ALL channels at once
                frame_rel_error = (total_error_sum / (total_gt_sum + 1e-10)) * 100.0
            else:
                # Fallback (physical space - not accurate)
                total_error_sum = 0.0
                total_gt_sum = 0.0
                for ivar in range(n_channels):
                    total_error_sum += np.abs(pred_sequence[frame, :, ivar] - gt_sequence[frame, :, ivar]).sum()
                    total_gt_sum += np.abs(gt_sequence[frame, :, ivar]).sum()
                frame_rel_error = (total_error_sum / (total_gt_sum + 1e-10)) * 100.0
            
            # Comment out per-frame prints (too noisy)
            # if frame < 10 or frame % 50 == 0:  # Print first 10 and every 50th frame
            #     print(f"Frame {frame:4d}: Mean relative error = {frame_rel_error:.2f}%")
        
        for ivar in range(n_channels):
            # Update ground truth
            scatter_objects['gt'][ivar].set_array(gt_sequence[frame, :, ivar])
            
            # Update prediction
            scatter_objects['pred'][ivar].set_array(pred_sequence[frame, :, ivar])
            
            # Update error - compute RELATIVE error in NORMALIZED space
            # For visualization, we show pointwise error, but computed in normalized space
            if show_error and scatter_objects['error'][ivar] is not None:
                gt_frame = gt_sequence[frame, :, ivar]
                pred_frame = pred_sequence[frame, :, ivar]
                
                # Normalize before computing relative error (critical!)
                if u_mean is not None and u_std is not None:
                    mean_ch = u_mean[0, ivar] if u_mean.ndim > 1 else u_mean[ivar]
                    std_ch = u_std[0, ivar] if u_std.ndim > 1 else u_std[ivar]
                    gt_norm = (gt_frame - mean_ch) / std_ch
                    pred_norm = (pred_frame - mean_ch) / std_ch
                    
                    # Pointwise relative error in normalized space for visualization
                    abs_err_norm = np.abs(pred_norm - gt_norm)
                    gt_abs_norm = np.abs(gt_norm)
                    relative_error = (abs_err_norm / (gt_abs_norm + 1e-10)) * 100.0
                else:
                    # Fallback (physical space - not good for multi-scale data)
                    numerator = np.abs(pred_frame - gt_frame)
                    denominator = np.abs(gt_frame) + 1e-10
                    relative_error = (numerator / denominator) * 100.0
                
                # Clamp error to 1-100% range for fixed colorscale
                relative_error_clamped = np.clip(relative_error, 1.0, 100.0)
                scatter_objects['error'][ivar].set_array(relative_error_clamped)
            
            # Dynamic colorscale update
            if dynamic_colorscale:
                # Get current frame data
                gt_frame = gt_sequence[frame, :, ivar]
                pred_frame = pred_sequence[frame, :, ivar]
                
                # Compute frame-specific range
                vmin_frame = min(gt_frame.min(), pred_frame.min())
                vmax_frame = max(gt_frame.max(), pred_frame.max())
                
                # Handle symmetric case
                if symmetric[ivar]:
                    abs_vmax_frame = max(abs(vmin_frame), abs(vmax_frame))
                    vmin_frame = -abs_vmax_frame
                    vmax_frame = abs_vmax_frame
                
                # Update colormaps (GT and Pred only, NOT error - it's fixed 1-100%)
                scatter_objects['gt'][ivar].set_clim(vmin=vmin_frame, vmax=vmax_frame)
                scatter_objects['pred'][ivar].set_clim(vmin=vmin_frame, vmax=vmax_frame)
                
                # Error colormap stays FIXED at 1-100% (do not update dynamically)
        
        # Update progress bar
        progress_bar[0].set_width(frame + 1)
        
        # Update text with time info
        if time_values and frame < len(time_values):
            current_time_ns = time_values[frame] * 1e9 if time_values[frame] < 1e-6 else time_values[frame]
            if total_duration_ns is not None:
                progress_text.set_text(f'Time: {current_time_ns:.2f} ns / {total_duration_ns:.2f} ns (frame {frame+1}/{n_timesteps})')
            else:
                progress_text.set_text(f'Time: {time_values[frame]:.3e} (frame {frame+1}/{n_timesteps})')
        else:
            progress_text.set_text(f'Frame: {frame+1}/{n_timesteps}')
        
        all_scatters = []
        for key in scatter_objects:
            all_scatters.extend([obj for obj in scatter_objects[key] if obj is not None])
        return all_scatters + [progress_bar[0], progress_text]
    
    anim = FuncAnimation(fig, animate, frames=n_timesteps, 
                        interval=interval, blit=False, repeat=True)
    
    print(f"Saving sequential animation to {save_path}...")
    try:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval, dpi=150)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=1000//interval, dpi=150)
        else:
            # Default to gif
            save_path_gif = save_path + '.gif'
            anim.save(save_path_gif, writer='pillow', fps=1000//interval, dpi=150)
            print(f"Animation saved as {save_path_gif}")
            return
        print(f"Sequential animation saved successfully: {save_path}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
        print("Try installing pillow (pip install pillow) for GIF support")
    
    plt.close(fig)


########################################################
# 1D Plotting functions
########################################################
def plot_estimates_1d(
    u_inp: np.ndarray,
    u_gtr: np.ndarray,
    u_prd: np.ndarray,
    x_inp: np.ndarray,
    x_out: np.ndarray,
    names: Optional[List[str]] = None,
    domain: Tuple[List[float], List[float]] = None,
    show_error: bool = True
) -> plt.Figure:
    """
    Plots input data, ground-truth, model predictions, and optionally absolute errors for 1D problems.
    
    Creates line plots instead of scatter plots for 1D spatial coordinates.
    
    Parameters
    ----------
    u_inp : np.ndarray
        Input data array of shape (N_inp, n_vars)
    u_gtr : np.ndarray
        Ground-truth data array of shape (N_out, n_vars)
    u_prd : np.ndarray
        Model-predicted data array of shape (N_out, n_vars)
    x_inp : np.ndarray
        1D coordinates of input points, shape (N_inp, 1)
    x_out : np.ndarray
        1D coordinates for output/ground-truth, shape (N_out, 1)
    names : list of str, optional
        Variable names for labels
    domain : tuple of list, optional
        Spatial domain ([x_min,], [x_max,])
    show_error : bool, optional
        Whether to show error plot
        
    Returns
    -------
    plt.Figure
        The generated matplotlib figure
    """
    # Extract 1D coordinates
    x_inp_1d = x_inp[:, 0] if x_inp.ndim > 1 else x_inp
    x_out_1d = x_out[:, 0] if x_out.ndim > 1 else x_out
    
    # Determine number of variables
    n_vars = u_gtr.shape[-1]
    
    # Generate names if not provided
    if names is None:
        names = [f"Variable {i:02d}" for i in range(n_vars)]
    
    # Create figure with subplots
    n_cols = 4 if show_error else 3
    fig, axes = plt.subplots(n_vars, n_cols, figsize=(4*n_cols, 2.5*n_vars), squeeze=False)
    
    col_titles = ['Input', 'Ground Truth', 'Prediction']
    if show_error:
        col_titles.append('Absolute Error')
    
    for ivar in range(n_vars):
        # Input
        ax = axes[ivar, 0]
        ax.plot(x_inp_1d, u_inp[:, ivar], 'b-', linewidth=1.5, label='Input')
        ax.set_ylabel(names[ivar], fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if ivar == 0:
            ax.set_title(col_titles[0], fontsize=11, fontweight='bold')
        if ivar == n_vars - 1:
            ax.set_xlabel('Position', fontsize=9)
        
        # Ground truth
        ax = axes[ivar, 1]
        ax.plot(x_out_1d, u_gtr[:, ivar], 'g-', linewidth=1.5, label='Ground Truth')
        ax.grid(True, alpha=0.3)
        if ivar == 0:
            ax.set_title(col_titles[1], fontsize=11, fontweight='bold')
        if ivar == n_vars - 1:
            ax.set_xlabel('Position', fontsize=9)
        
        # Prediction
        ax = axes[ivar, 2]
        ax.plot(x_out_1d, u_prd[:, ivar], 'r-', linewidth=1.5, label='Prediction')
        ax.grid(True, alpha=0.3)
        if ivar == 0:
            ax.set_title(col_titles[2], fontsize=11, fontweight='bold')
        if ivar == n_vars - 1:
            ax.set_xlabel('Position', fontsize=9)
        
        # Error
        if show_error:
            ax = axes[ivar, 3]
            error = np.abs(u_gtr[:, ivar] - u_prd[:, ivar])
            ax.plot(x_out_1d, error, 'k-', linewidth=1.5, label='|GT - Pred|')
            ax.grid(True, alpha=0.3)
            if ivar == 0:
                ax.set_title(col_titles[3], fontsize=11, fontweight='bold')
            if ivar == n_vars - 1:
                ax.set_xlabel('Position', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_sequential_animation_1d(
    gt_sequence: np.ndarray,
    pred_sequence: np.ndarray,
    coords: np.ndarray,
    save_path: str,
    input_data: np.ndarray = None,
    time_values: List[float] = None,
    interval: int = 500,
    names: Optional[List[str]] = None,
    domain: Tuple[List[float], List[float]] = None,
    show_error: bool = True,
    u_mean: np.ndarray = None,
    u_std: np.ndarray = None,
    max_frames: int = 300
):
    """
    Create animated line plots for 1D sequential data.
    
    Parameters
    ----------
    gt_sequence : np.ndarray
        Ground truth sequence, shape (T, N, C)
    pred_sequence : np.ndarray
        Predicted sequence, shape (T, N, C)
    coords : np.ndarray
        1D spatial coordinates, shape (N, 1)
    save_path : str
        Path to save animation
    input_data : np.ndarray, optional
        Initial input data
    time_values : list of float, optional
        Time values for each frame
    interval : int
        Delay between frames in milliseconds
    names : list of str, optional
        Variable names
    domain : tuple, optional
        Spatial domain
    show_error : bool
        Whether to show error subplot
    u_mean : np.ndarray, optional
        Mean for denormalization
    u_std : np.ndarray, optional
        Std for denormalization
    max_frames : int, optional
        Maximum number of frames to include (downsamples if necessary)
    """
    import matplotlib.animation as animation
    
    # Extract 1D coordinates
    x_coords = coords[:, 0] if coords.ndim > 1 else coords
    
    T, N, n_channels = gt_sequence.shape
    
    # Downsample frames if necessary to avoid memory issues
    if T > max_frames:
        print(f"[Animation] Downsampling from {T} to {max_frames} frames to reduce memory usage")
        frame_indices = np.linspace(0, T-1, max_frames, dtype=int)
        gt_sequence = gt_sequence[frame_indices]
        pred_sequence = pred_sequence[frame_indices]
        if time_values is not None:
            time_values = [time_values[i] for i in frame_indices]
        T = max_frames
    else:
        frame_indices = np.arange(T)
    
    # Generate names if not provided
    if names is None:
        names = [f"Variable {i:02d}" for i in range(n_channels)]
    
    # Compute global value ranges for consistent y-axis
    gt_min = gt_sequence.min(axis=(0, 1))  # [C]
    gt_max = gt_sequence.max(axis=(0, 1))  # [C]
    pred_min = pred_sequence.min(axis=(0, 1))
    pred_max = pred_sequence.max(axis=(0, 1))
    
    val_min = np.minimum(gt_min, pred_min)
    val_max = np.maximum(gt_max, pred_max)
    
    # Add margin
    margin = 0.1 * (val_max - val_min)
    val_min -= margin
    val_max += margin
    
    # Create figure
    n_cols = 3 if show_error else 2
    fig, axes = plt.subplots(n_channels, n_cols, figsize=(6*n_cols, 2.5*n_channels), squeeze=False)
    
    # Initialize line objects
    lines = {'gt': [], 'pred': [], 'error': []}
    
    col_titles = ['Ground Truth', 'Prediction']
    if show_error:
        col_titles.append('Absolute Error')
    
    for ivar in range(n_channels):
        # Ground truth subplot
        ax_gt = axes[ivar, 0]
        line_gt, = ax_gt.plot(x_coords, gt_sequence[0, :, ivar], 'g-', linewidth=2)
        lines['gt'].append(line_gt)
        ax_gt.set_ylim(val_min[ivar], val_max[ivar])
        ax_gt.set_ylabel(names[ivar], fontsize=10, fontweight='bold')
        ax_gt.grid(True, alpha=0.3)
        if ivar == 0:
            ax_gt.set_title(col_titles[0], fontsize=11, fontweight='bold')
        if ivar == n_channels - 1:
            ax_gt.set_xlabel('Position', fontsize=9)
        
        # Prediction subplot
        ax_pred = axes[ivar, 1]
        line_pred, = ax_pred.plot(x_coords, pred_sequence[0, :, ivar], 'r-', linewidth=2)
        lines['pred'].append(line_pred)
        ax_pred.set_ylim(val_min[ivar], val_max[ivar])
        ax_pred.grid(True, alpha=0.3)
        if ivar == 0:
            ax_pred.set_title(col_titles[1], fontsize=11, fontweight='bold')
        if ivar == n_channels - 1:
            ax_pred.set_xlabel('Position', fontsize=9)
        
        # Error subplot
        if show_error:
            ax_err = axes[ivar, 2]
            error_0 = np.abs(gt_sequence[0, :, ivar] - pred_sequence[0, :, ivar])
            line_err, = ax_err.plot(x_coords, error_0, 'k-', linewidth=2)
            lines['error'].append(line_err)
            # Set error y-axis range
            max_err = np.abs(gt_sequence[:, :, ivar] - pred_sequence[:, :, ivar]).max()
            ax_err.set_ylim(0, max_err * 1.1)
            ax_err.grid(True, alpha=0.3)
            if ivar == 0:
                ax_err.set_title(col_titles[2], fontsize=11, fontweight='bold')
            if ivar == n_channels - 1:
                ax_err.set_xlabel('Position', fontsize=9)
    
    # Add time text
    time_text = fig.suptitle('', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    def update_frame(frame):
        """Update function for animation"""
        # Update title with time
        if time_values is not None and frame < len(time_values):
            time_text.set_text(f'Time: {time_values[frame]:.6e} s')
        else:
            time_text.set_text(f'Frame: {frame}')
        
        # Update each variable
        for ivar in range(n_channels):
            # Update ground truth
            lines['gt'][ivar].set_ydata(gt_sequence[frame, :, ivar])
            
            # Update prediction
            lines['pred'][ivar].set_ydata(pred_sequence[frame, :, ivar])
            
            # Update error
            if show_error:
                error = np.abs(gt_sequence[frame, :, ivar] - pred_sequence[frame, :, ivar])
                lines['error'][ivar].set_ydata(error)
        
        return [time_text] + lines['gt'] + lines['pred'] + (lines['error'] if show_error else [])
    
    # Create animation with memory-efficient settings
    anim = animation.FuncAnimation(
        fig, update_frame, frames=T,
        interval=interval, blit=True, repeat=True
    )
    
    # Save animation with error handling
    try:
        # Use lower DPI for 1D to save memory (line plots don't need high resolution)
        anim.save(save_path, writer='pillow', fps=1000//interval, dpi=100)
        print(f"1D animation saved successfully: {save_path} ({T} frames)")
    except MemoryError as e:
        print(f"MemoryError: Animation too large. Try reducing max_frames (current: {T})")
        print(f"Suggestion: Set max_frames to ~100-200 for your dataset size")
        plt.close(fig)
        raise
    except Exception as e:
        print(f"Failed to save animation: {e}")
        print("Try installing pillow (pip install pillow) for GIF support")
    
    plt.close(fig)
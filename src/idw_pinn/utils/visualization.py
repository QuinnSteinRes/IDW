"""
Visualization utilities for 2D diffusion inverse problems.

Provides publication-ready plots with:
- Large fonts/axes for Overleaf/presentations
- Unique identifiers (config params + datetime) in filenames
- Whole figures + separate subfigures for LaTeX integration
- Support for numerical (with true D=0.2) and experimental data (target D range)
- Percent error display instead of absolute error
- Consistent axis limits across runs for comparison

Updated: January 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os
from datetime import datetime
import uuid


# =============================================================================
# DATA TYPE SELECTION - CHANGE THIS FLAG
# =============================================================================

DATA_TYPE = 'experimental'  # OPTIONS: 'numerical' or 'experimental'


# =============================================================================
# PUBLICATION-READY DEFAULTS
# =============================================================================

FONT_CONFIG = {
    'title': 16,
    'suptitle': 18,
    'axis_label': 14,
    'tick_label': 12,
    'colorbar_label': 13,
    'legend': 12,
}

# Hardcoded axis limits for cross-run consistency (all log scale)
AXIS_LIMITS = {
    'D_evolution': (1e-5, 1),
    'D_error': (1e-10, 1),
    'losses': (1e-12, 1e2),
    'lambdas': (1e-3, 1e5),
}

DPI_SAVE = 300

# Data type configurations
DATA_CONFIG = {
    'numerical': {
        'diff_coeff_true': 0.2,
        'diff_coeff_display': '0.2',
        'label': 'True D',
    },
    'experimental': {
        # Target D range in m²/s: 3.15e-10 to 4.05e-10
        # Corresponding D_norm range: 0.000507 to 0.000652
        'diff_coeff_range': (0.000507, 0.000652),
        'diff_coeff_physical_range': (3.15e-10, 4.05e-10),  # m²/s
        'label': 'Target D range',
    }
}


def set_publication_style():
    """Set matplotlib defaults for publication-quality figures."""
    plt.rcParams.update({
        'font.size': FONT_CONFIG['axis_label'],
        'axes.titlesize': FONT_CONFIG['title'],
        'axes.labelsize': FONT_CONFIG['axis_label'],
        'xtick.labelsize': FONT_CONFIG['tick_label'],
        'ytick.labelsize': FONT_CONFIG['tick_label'],
        'legend.fontsize': FONT_CONFIG['legend'],
        'figure.titlesize': FONT_CONFIG['suptitle'],
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.5,
        'savefig.dpi': DPI_SAVE,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


# =============================================================================
# FILENAME GENERATION
# =============================================================================

def generate_unique_filename(base_name: str, extension: str = 'png', 
                             diff_coeff: float = None) -> str:
    """
    Generate unique filename with timestamp, UUID, and optionally D value.
    
    Format: {base_name}_D_{diff_coeff}_{timestamp}_{uuid}.{extension}
    """
    timestamp = datetime.now().strftime('%d%m%y_%H%M%S')
    short_uuid = uuid.uuid4().hex[:4]
    
    if diff_coeff is not None:
        d_str = f"D_{diff_coeff:.6f}"
        return f"{base_name}_{d_str}_{timestamp}_{short_uuid}.{extension}"
    else:
        return f"{base_name}_{timestamp}_{short_uuid}.{extension}"


def _ensure_dirs(output_dir: str):
    """Ensure output and subfigures directories exist."""
    os.makedirs(output_dir, exist_ok=True)
    subfig_dir = os.path.join(output_dir, 'subfigures')
    os.makedirs(subfig_dir, exist_ok=True)
    return subfig_dir


# =============================================================================
# LATEX SNIPPET GENERATION
# =============================================================================

def _append_latex_snippet(output_dir: str, subfigure_paths: list, 
                          figure_type: str, caption: str = '', label: str = ''):
    """
    Append LaTeX snippet for subfigures to latex_snippets.txt.
    
    Args:
        output_dir: Directory containing the output
        subfigure_paths: List of paths to subfigure PDFs
        figure_type: Type of figure (e.g., 'solution_comparison', 'diagnostics')
        caption: Figure caption text
        label: LaTeX label for the figure
    """
    latex_file = os.path.join(output_dir, 'latex_snippets.txt')
    
    # Extract just filenames (relative to Figs/ folder in Overleaf)
    filenames = [os.path.basename(p) for p in subfigure_paths]
    
    # Determine number of columns (4 for standard layout)
    n_cols = min(4, len(filenames))
    
    # Generate timestamp and unique run ID for this entry
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    
    # Build LaTeX snippet
    tabular_cols = 'c@{}' * n_cols
    snippet = f"""
% =============================================================================
% {figure_type.upper()} - Generated: {timestamp}
% OVERLEAF DIRECTORY: Figs/{run_id}
% Create this directory in Overleaf and upload all subfigures there.
% =============================================================================
\\begin{{figure}}[htbp]
\\checkoddpage
\\begin{{adjustwidth}}{{-1cm}}{{-1cm}}
\\centering
\\setlength{{\\tabcolsep}}{{0pt}}
\\begin{{tabular}}{{@{{}}{tabular_cols}}}
"""
    
    # Add images row by row
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', 
              '(i)', '(j)', '(k)', '(l)']
    
    for row_start in range(0, len(filenames), n_cols):
        row_files = filenames[row_start:row_start + n_cols]
        row_labels = labels[row_start:row_start + n_cols]
        
        # Image row - use run_id directory
        img_commands = [f'\\includegraphics[width={0.95/n_cols:.2f}\\textwidth]{{Figs/{run_id}/{f}}}' 
                       for f in row_files]
        snippet += '    ' + ' &\n    '.join(img_commands) + r' \\' + '\n'
        
        # Label row
        label_commands = [f'\\small {lbl}' for lbl in row_labels]
        snippet += '    ' + ' & '.join(label_commands) + r' \\[0.5em]' + '\n'
    
    snippet += f"""\\end{{tabular}}
\\end{{adjustwidth}}
\\caption{{{caption if caption else f'{figure_type} results'}}}
\\label{{{label if label else f'fig:{figure_type}'}}}
\\end{{figure}}

% Upload these files to Figs/{run_id}/:
"""
    for p in subfigure_paths:
        snippet += f"%   {os.path.basename(p)}\n"
    
    snippet += "\n"
    
    # Append to file
    with open(latex_file, 'a') as f:
        f.write(snippet)
    
    print(f"LaTeX snippet appended to: {latex_file}")
    print(f">>> Create Overleaf directory: Figs/{run_id}")


# =============================================================================
# SOLUTION COMPARISON PLOTS
# =============================================================================

def plot_2d_solution_comparison(u_pred, usol, x, y, t, diff_coeff_learned,
                                 diff_coeff_true=None, data_type=None,
                                 output_dir='outputs', filename=None,
                                 save_subfigures=True):
    # Use global DATA_TYPE if not specified
    if data_type is None:
        data_type = DATA_TYPE
    """
    Plot 2D solution comparison at 4 time slices with PERCENT ERROR.
    
    Creates a 3-row figure:
    - Row 0: True/measured solution at 4 time points
    - Row 1: Predicted solution at 4 time points
    - Row 2: Percent error at 4 time points
    
    Args:
        u_pred: Predicted solution, shape (Nx+1, Ny+1, Nt) or flattened
        usol: True/measured solution, shape (Nx+1, Ny+1, Nt)
        x: x coordinates (Nx+1,)
        y: y coordinates (Ny+1,)
        t: time coordinates (Nt,)
        diff_coeff_learned: Learned diffusion coefficient
        diff_coeff_true: True D (for numerical) or None (for experimental)
        data_type: 'numerical' or 'experimental'
        output_dir: Directory to save figures
        filename: Output filename (if None, generates unique name)
        save_subfigures: Whether to save individual subfigures for LaTeX
    
    Returns:
        dict: Paths to saved files {'whole': path, 'subfigures': [paths]}
    """
    set_publication_style()
    subfig_dir = _ensure_dirs(output_dir)
    
    # Generate unique filename if not provided
    if filename is None:
        filename = generate_unique_filename('solution_comparison', 'png', diff_coeff_learned)
    
    # Reshape prediction to match solution shape if needed
    if u_pred.ndim == 2 and u_pred.shape[1] == 1:
        u_pred = u_pred.flatten()
    
    if u_pred.ndim == 1:
        u_pred_reshaped = np.reshape(u_pred, usol.shape, order='C')
    else:
        u_pred_reshaped = u_pred
    
    # Select 4 time indices for plotting
    nt = len(t)
    t_indices = [0, nt//3, 2*nt//3, nt-1]
    n_cols = len(t_indices)
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Compute global colorbar limits for solution
    u_vmin = min(usol.min(), u_pred_reshaped.min())
    u_vmax = max(usol.max(), u_pred_reshaped.max())
    
    # Compute PERCENT ERROR
    # Avoid division by zero: use max(|true|, small_value) as denominator
    usol_abs_max = np.maximum(np.abs(usol), 1e-10)
    percent_error_all = 100.0 * np.abs(usol - u_pred_reshaped) / usol_abs_max
    
    # Cap percent error for visualization (very small true values cause huge %)
    percent_error_all = np.clip(percent_error_all, 0, 100)
    err_vmax = min(percent_error_all.max(), 50)  # Cap at 50% for colorbar
    
    # Determine labels based on data type
    if data_type == 'numerical':
        true_label = 'True'
        d_display = f'True D={DATA_CONFIG["numerical"]["diff_coeff_true"]}'
    else:
        true_label = 'Measured'
        d_range = DATA_CONFIG['experimental']['diff_coeff_range']
        d_display = f'Target D: {d_range[0]:.4f}-{d_range[1]:.4f}'
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(18, 13))
    gs = gridspec.GridSpec(3, n_cols + 1, width_ratios=[1, 1, 1, 1, 0.05],
                           wspace=0.25, hspace=0.3)
    
    saved_subfigures = []
    
    # Row 0: True/measured solution
    axes_true = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        im0 = axes_true[i].pcolormesh(X, Y, usol[:, :, ti], cmap='jet',
                                       shading='auto', vmin=u_vmin, vmax=u_vmax)
        axes_true[i].set_title(f'{true_label}, t={t[ti]:.3f}s', fontsize=FONT_CONFIG['title'])
        axes_true[i].set_xlabel('x', fontsize=FONT_CONFIG['axis_label'])
        if i == 0:
            axes_true[i].set_ylabel('y', fontsize=FONT_CONFIG['axis_label'])
        axes_true[i].set_aspect('equal')
        axes_true[i].tick_params(labelsize=FONT_CONFIG['tick_label'])
        
        # Save subfigure
        if save_subfigures:
            subfig_path = os.path.join(subfig_dir, 
                f'{true_label.lower()}_t{i}_D{diff_coeff_learned:.6f}.pdf')
            _save_single_panel(X, Y, usol[:, :, ti], 'jet', u_vmin, u_vmax,
                              f'{true_label}, t={t[ti]:.3f}s', 'u', subfig_path)
            saved_subfigures.append(subfig_path)
    
    cax0 = fig.add_subplot(gs[0, n_cols])
    cbar0 = fig.colorbar(im0, cax=cax0)
    cbar0.set_label(f'u ({true_label})', fontsize=FONT_CONFIG['colorbar_label'])
    cbar0.ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Row 1: Predicted solution
    axes_pred = [fig.add_subplot(gs[1, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        im1 = axes_pred[i].pcolormesh(X, Y, u_pred_reshaped[:, :, ti], cmap='jet',
                                       shading='auto', vmin=u_vmin, vmax=u_vmax)
        axes_pred[i].set_title(f'Predicted, t={t[ti]:.3f}s', fontsize=FONT_CONFIG['title'])
        axes_pred[i].set_xlabel('x', fontsize=FONT_CONFIG['axis_label'])
        if i == 0:
            axes_pred[i].set_ylabel('y', fontsize=FONT_CONFIG['axis_label'])
        axes_pred[i].set_aspect('equal')
        axes_pred[i].tick_params(labelsize=FONT_CONFIG['tick_label'])
        
        # Save subfigure
        if save_subfigures:
            subfig_path = os.path.join(subfig_dir,
                f'pred_t{i}_D{diff_coeff_learned:.6f}.pdf')
            _save_single_panel(X, Y, u_pred_reshaped[:, :, ti], 'jet', u_vmin, u_vmax,
                              f'Predicted, t={t[ti]:.3f}s', 'u', subfig_path)
            saved_subfigures.append(subfig_path)
    
    cax1 = fig.add_subplot(gs[1, n_cols])
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('u (Pred)', fontsize=FONT_CONFIG['colorbar_label'])
    cbar1.ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Row 2: Percent Error
    axes_err = [fig.add_subplot(gs[2, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        im2 = axes_err[i].pcolormesh(X, Y, percent_error_all[:, :, ti], cmap='hot',
                                      shading='auto', vmin=0, vmax=err_vmax)
        axes_err[i].set_title(f'% Error, t={t[ti]:.3f}s', fontsize=FONT_CONFIG['title'])
        axes_err[i].set_xlabel('x', fontsize=FONT_CONFIG['axis_label'])
        if i == 0:
            axes_err[i].set_ylabel('y', fontsize=FONT_CONFIG['axis_label'])
        axes_err[i].set_aspect('equal')
        axes_err[i].tick_params(labelsize=FONT_CONFIG['tick_label'])
        
        # Save subfigure
        if save_subfigures:
            subfig_path = os.path.join(subfig_dir,
                f'error_t{i}_D{diff_coeff_learned:.6f}.pdf')
            _save_single_panel(X, Y, percent_error_all[:, :, ti], 'hot', 0, err_vmax,
                              f'% Error, t={t[ti]:.3f}s', '% Error', subfig_path)
            saved_subfigures.append(subfig_path)
    
    cax2 = fig.add_subplot(gs[2, n_cols])
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('% Error', fontsize=FONT_CONFIG['colorbar_label'])
    cbar2.ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Title with appropriate D display
    plt.suptitle(f'2D Diffusion: Learned D = {diff_coeff_learned:.6f} ({d_display})',
                 fontsize=FONT_CONFIG['suptitle'])
    
    # Save as PDF only
    filepath = os.path.join(output_dir, filename.replace('.png', '.pdf'))
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()
    
    # Generate LaTeX snippet
    if save_subfigures and saved_subfigures:
        _append_latex_snippet(
            output_dir=output_dir,
            subfigure_paths=saved_subfigures,
            figure_type='solution_comparison',
            caption=f'2D diffusion solution comparison. Learned D = {diff_coeff_learned:.6f}.',
            label='fig:solution_comparison'
        )
    
    return {'whole': filepath, 'subfigures': saved_subfigures}


def _save_single_panel(X, Y, data, cmap, vmin, vmax, title, cbar_label, filepath):
    """Save a single panel as a standalone figure for LaTeX."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.pcolormesh(X, Y, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=FONT_CONFIG['title'])
    ax.set_xlabel('x', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('y', fontsize=FONT_CONFIG['axis_label'])
    ax.set_aspect('equal')
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=FONT_CONFIG['colorbar_label'])
    cbar.ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close()


# =============================================================================
# TRAINING DIAGNOSTICS
# =============================================================================

def plot_training_diagnostics(history_adam, history_lbfgs, diff_coeff_true=None,
                               data_type=None, output_dir='outputs',
                               filename=None, save_subfigures=True):
    """
    Plot comprehensive training diagnostics in 2x3 layout (all log scale).
    
    Creates diagnostic plots showing:
    - D evolution (log scale) x2
    - D error evolution (log scale)
    - Loss components (BC/IC, Data, PDE) over epochs (log scale)
    - Lambda weights (log scale) x2
    
    All plots use consistent hardcoded axis limits from AXIS_LIMITS for
    cross-run comparison.
    
    Args:
        history_adam: Dict with keys 'diff_coeff', 'loss_bc', 'loss_data', 'loss_f',
                      'lam_bc', 'lam_data', 'lam_f' from Adam phase
        history_lbfgs: Dict with same keys from L-BFGS phase
        diff_coeff_true: True D for numerical data, or None for experimental
        data_type: 'numerical' or 'experimental'
        output_dir: Directory to save figure
        filename: Output filename (if None, generates unique name)
        save_subfigures: Whether to save individual subfigures for LaTeX
    
    Returns:
        dict: Paths to saved files
    """
    # Use global DATA_TYPE if not specified
    if data_type is None:
        data_type = DATA_TYPE
        
    set_publication_style()
    subfig_dir = _ensure_dirs(output_dir)
    
    # Generate unique filename if not provided
    if filename is None:
        final_D = history_lbfgs.get('diff_coeff', history_adam['diff_coeff'])[-1] \
                  if history_lbfgs.get('diff_coeff') else history_adam['diff_coeff'][-1]
        filename = generate_unique_filename('training_diagnostics', 'png', final_D)
    
    # Combine histories
    adam_epochs = len(history_adam['diff_coeff'])
    
    diff_all = list(history_adam['diff_coeff']) + list(history_lbfgs.get('diff_coeff', []))
    loss_bc_all = list(history_adam['loss_bc']) + list(history_lbfgs.get('loss_bc', []))
    loss_data_all = list(history_adam['loss_data']) + list(history_lbfgs.get('loss_data', []))
    loss_f_all = list(history_adam['loss_f']) + list(history_lbfgs.get('loss_f', []))
    lam_bc_all = list(history_adam['lam_bc']) + list(history_lbfgs.get('lam_bc', []))
    lam_data_all = list(history_adam['lam_data']) + list(history_lbfgs.get('lam_data', []))
    lam_f_all = list(history_adam['lam_f']) + list(history_lbfgs.get('lam_f', []))
    
    total_iterations = len(diff_all)
    
    # Determine D reference display
    if data_type == 'numerical':
        d_true = diff_coeff_true if diff_coeff_true is not None else DATA_CONFIG['numerical']['diff_coeff_true']
        d_label = f'True D={d_true}'
        d_range = None
    else:
        d_range = DATA_CONFIG['experimental']['diff_coeff_range']
        d_label = f'Target: {d_range[0]:.4f}-{d_range[1]:.4f}'
        d_true = None
    
    # Create 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    saved_subfigures = []
    
    # -------------------------------------------------------------------------
    # Plot 1: D evolution (log scale)
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.semilogy(range(adam_epochs), history_adam['diff_coeff'], 'b-', alpha=0.8, linewidth=1.5, label='Adam')
    if history_lbfgs.get('diff_coeff'):
        ax.semilogy(range(adam_epochs, total_iterations), history_lbfgs['diff_coeff'],
                'b--', alpha=0.8, linewidth=1.5, label='L-BFGS')
    
    # Reference line(s)
    if d_true is not None:
        ax.axhline(y=d_true, color='r', linestyle='-', linewidth=2, label=d_label)
    elif d_range is not None:
        ax.axhspan(d_range[0], d_range[1], alpha=0.3, color='green', label=d_label)
    
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.set_ylim(AXIS_LIMITS['D_evolution'])
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('D (log)', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('Diffusion Coefficient Evolution', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        path = os.path.join(subfig_dir, 'D_evolution_1.pdf')
        _save_diagnostic_subplot_direct(
            x_data=[(range(adam_epochs), history_adam['diff_coeff'], 'b-', 'Adam'),
                    (range(adam_epochs, total_iterations), history_lbfgs.get('diff_coeff', []), 'b--', 'L-BFGS')],
            ylabel='D (log)', title='Diffusion Coefficient Evolution',
            ylim=AXIS_LIMITS['D_evolution'], adam_epochs=adam_epochs,
            d_true=d_true, d_range=d_range, d_label=d_label,
            filepath=path, use_log=True
        )
        saved_subfigures.append(path)
    
    # -------------------------------------------------------------------------
    # Plot 2: D evolution (log scale) - duplicate for layout consistency
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    ax.semilogy(range(adam_epochs), history_adam['diff_coeff'], 'b-', alpha=0.8, linewidth=1.5)
    if history_lbfgs.get('diff_coeff'):
        ax.semilogy(range(adam_epochs, total_iterations), history_lbfgs['diff_coeff'],
                    'b--', alpha=0.8, linewidth=1.5)
    
    if d_true is not None:
        ax.axhline(y=d_true, color='r', linestyle='-', linewidth=2, label=d_label)
    elif d_range is not None:
        ax.axhspan(d_range[0], d_range[1], alpha=0.3, color='green', label=d_label)
    
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.set_ylim(AXIS_LIMITS['D_evolution'])
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('D (log)', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('D Evolution (Log Scale)', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        path = os.path.join(subfig_dir, 'D_evolution_2.pdf')
        _save_diagnostic_subplot_direct(
            x_data=[(range(adam_epochs), history_adam['diff_coeff'], 'b-', 'Adam'),
                    (range(adam_epochs, total_iterations), history_lbfgs.get('diff_coeff', []), 'b--', 'L-BFGS')],
            ylabel='D (log)', title='D Evolution (Log Scale)',
            ylim=AXIS_LIMITS['D_evolution'], adam_epochs=adam_epochs,
            d_true=d_true, d_range=d_range, d_label=d_label,
            filepath=path, use_log=True
        )
        saved_subfigures.append(path)
    
    # -------------------------------------------------------------------------
    # Plot 3: D error evolution (numerical) or % error from midpoint (experimental)
    # -------------------------------------------------------------------------
    ax = axes[0, 2]
    
    if data_type == 'numerical' and d_true is not None:
        d_error = np.abs(np.array(diff_all) - d_true)
        ax.semilogy(range(adam_epochs), d_error[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
        if len(d_error) > adam_epochs:
            ax.semilogy(range(adam_epochs, total_iterations), d_error[adam_epochs:],
                        'b--', alpha=0.8, linewidth=1.5)
        ylabel = '|D - D_true|'
        title = 'D Error Evolution'
        error_data = d_error
    else:
        # For experimental: percent error from target range midpoint
        d_arr = np.array(diff_all)
        d_low, d_high = d_range
        d_midpoint = (d_low + d_high) / 2.0
        percent_error = np.abs(d_arr - d_midpoint) / d_midpoint * 100.0
        ax.semilogy(range(adam_epochs), percent_error[:adam_epochs] + 1e-12, 'b-', alpha=0.8, linewidth=1.5)
        if len(percent_error) > adam_epochs:
            ax.semilogy(range(adam_epochs, total_iterations), percent_error[adam_epochs:] + 1e-12,
                        'b--', alpha=0.8, linewidth=1.5)
        ylabel = '% Error from midpoint'
        title = 'D % Error from Target Midpoint'
        error_data = percent_error + 1e-12
    
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.set_ylim(AXIS_LIMITS['D_error'])
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel(ylabel, fontsize=FONT_CONFIG['axis_label'])
    ax.set_title(title, fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        path = os.path.join(subfig_dir, 'D_error.pdf')
        _save_diagnostic_subplot_direct(
            x_data=[(range(adam_epochs), error_data[:adam_epochs], 'b-', 'Adam'),
                    (range(adam_epochs, total_iterations), error_data[adam_epochs:] if len(error_data) > adam_epochs else [], 'b--', 'L-BFGS')],
            ylabel=ylabel, title=title,
            ylim=AXIS_LIMITS['D_error'], adam_epochs=adam_epochs,
            d_true=None, d_range=None, d_label=None,
            filepath=path, use_log=True
        )
        saved_subfigures.append(path)
    
    # -------------------------------------------------------------------------
    # Plot 4: Loss components
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    ax.semilogy(range(adam_epochs), loss_bc_all[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), loss_data_all[:adam_epochs], 'orange', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), loss_f_all[:adam_epochs], 'g-', alpha=0.8, linewidth=1.5)
    if history_lbfgs.get('loss_bc'):
        ax.semilogy(range(adam_epochs, total_iterations), loss_bc_all[adam_epochs:],
                    'b--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), loss_data_all[adam_epochs:],
                    'orange', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), loss_f_all[adam_epochs:],
                    'g--', alpha=0.8, linewidth=1.5)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', linewidth=2, label=r'$\mathcal{L}_{BC/IC}$')
    ax.plot([], [], 'orange', linewidth=2, label=r'$\mathcal{L}_{data}$')
    ax.plot([], [], 'g-', linewidth=2, label=r'$\mathcal{L}_{PDE}$')
    ax.set_ylim(AXIS_LIMITS['losses'])
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Loss', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('Loss Components', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        path = os.path.join(subfig_dir, 'losses.pdf')
        _save_loss_subplot(
            adam_epochs=adam_epochs, total_iterations=total_iterations,
            loss_bc=loss_bc_all, loss_data=loss_data_all, loss_f=loss_f_all,
            ylabel='Loss', title='Loss Components',
            ylim=AXIS_LIMITS['losses'], filepath=path
        )
        saved_subfigures.append(path)
    
    # -------------------------------------------------------------------------
    # Plot 5: Lambda weights (log scale)
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.semilogy(range(adam_epochs), lam_bc_all[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), lam_data_all[:adam_epochs], 'orange', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), lam_f_all[:adam_epochs], 'g-', alpha=0.8, linewidth=1.5)
    if history_lbfgs.get('lam_bc'):
        ax.semilogy(range(adam_epochs, total_iterations), lam_bc_all[adam_epochs:],
                'b--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), lam_data_all[adam_epochs:],
                'orange', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), lam_f_all[adam_epochs:],
                'g--', alpha=0.8, linewidth=1.5)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', linewidth=2, label=r'$\lambda_{BC/IC}$')
    ax.plot([], [], 'orange', linewidth=2, label=r'$\lambda_{data}$')
    ax.plot([], [], 'g-', linewidth=2, label=r'$\lambda_{PDE}$')
    ax.set_ylim(AXIS_LIMITS['lambdas'])
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Weight (log)', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('IDW Weights', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        path = os.path.join(subfig_dir, 'lambdas_1.pdf')
        _save_lambda_subplot(
            adam_epochs=adam_epochs, total_iterations=total_iterations,
            lam_bc=lam_bc_all, lam_data=lam_data_all, lam_f=lam_f_all,
            ylabel='Weight (log)', title='IDW Weights',
            ylim=AXIS_LIMITS['lambdas'], filepath=path
        )
        saved_subfigures.append(path)
    
    # -------------------------------------------------------------------------
    # Plot 6: Lambda weights (log scale) - duplicate for layout
    # -------------------------------------------------------------------------
    ax = axes[1, 2]
    ax.semilogy(range(adam_epochs), lam_bc_all[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), lam_data_all[:adam_epochs], 'orange', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), lam_f_all[:adam_epochs], 'g-', alpha=0.8, linewidth=1.5)
    if history_lbfgs.get('lam_bc'):
        ax.semilogy(range(adam_epochs, total_iterations), lam_bc_all[adam_epochs:],
                    'b--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), lam_data_all[adam_epochs:],
                    'orange', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), lam_f_all[adam_epochs:],
                    'g--', alpha=0.8, linewidth=1.5)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', linewidth=2, label=r'$\lambda_{BC/IC}$')
    ax.plot([], [], 'orange', linewidth=2, label=r'$\lambda_{data}$')
    ax.plot([], [], 'g-', linewidth=2, label=r'$\lambda_{PDE}$')
    ax.set_ylim(AXIS_LIMITS['lambdas'])
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Weight (log)', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('IDW Weights (Log Scale)', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        path = os.path.join(subfig_dir, 'lambdas_2.pdf')
        _save_lambda_subplot(
            adam_epochs=adam_epochs, total_iterations=total_iterations,
            lam_bc=lam_bc_all, lam_data=lam_data_all, lam_f=lam_f_all,
            ylabel='Weight (log)', title='IDW Weights (Log Scale)',
            ylim=AXIS_LIMITS['lambdas'], filepath=path
        )
        saved_subfigures.append(path)
    
    plt.tight_layout()
    
    # Save as PDF only
    filepath = os.path.join(output_dir, filename.replace('.png', '.pdf'))
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()
    
    # Generate LaTeX snippet
    if save_subfigures and saved_subfigures:
        _append_latex_snippet(
            output_dir=output_dir,
            subfigure_paths=saved_subfigures,
            figure_type='training_diagnostics',
            caption='Training diagnostics showing D evolution, error, losses, and IDW weights.',
            label='fig:training_diagnostics'
        )
    
    return {'whole': filepath, 'subfigures': saved_subfigures}


def _save_diagnostic_subplot_direct(x_data, ylabel, title, ylim, adam_epochs,
                                     d_true, d_range, d_label, filepath, use_log=True):
    """
    Save a diagnostic subplot with consistent axis limits.
    
    Args:
        x_data: List of tuples (x_vals, y_vals, linestyle, label)
        ylabel: Y-axis label
        title: Plot title
        ylim: (ymin, ymax) tuple
        adam_epochs: Iteration count for Adam phase (for vertical line)
        d_true: True D value (for horizontal line) or None
        d_range: (d_low, d_high) tuple for experimental data or None
        d_label: Label for D reference
        filepath: Output path
        use_log: Whether to use log scale on y-axis
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for x_vals, y_vals, linestyle, label in x_data:
        if len(y_vals) > 0:
            if use_log:
                ax.semilogy(x_vals, y_vals, linestyle, alpha=0.8, linewidth=1.5, label=label)
            else:
                ax.plot(x_vals, y_vals, linestyle, alpha=0.8, linewidth=1.5, label=label)
    
    # Add reference lines
    if d_true is not None:
        ax.axhline(y=d_true, color='r', linestyle='-', linewidth=2, label=d_label)
    elif d_range is not None:
        ax.axhspan(d_range[0], d_range[1], alpha=0.3, color='green', label=d_label)
    
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    
    ax.set_ylim(ylim)
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel(ylabel, fontsize=FONT_CONFIG['axis_label'])
    ax.set_title(title, fontsize=FONT_CONFIG['title'])
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_CONFIG['legend'])
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close()


def _save_loss_subplot(adam_epochs, total_iterations, loss_bc, loss_data, loss_f,
                       ylabel, title, ylim, filepath):
    """Save loss components subplot with consistent axis limits."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.semilogy(range(adam_epochs), loss_bc[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), loss_data[:adam_epochs], 'orange', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), loss_f[:adam_epochs], 'g-', alpha=0.8, linewidth=1.5)
    
    if len(loss_bc) > adam_epochs:
        ax.semilogy(range(adam_epochs, total_iterations), loss_bc[adam_epochs:],
                    'b--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), loss_data[adam_epochs:],
                    'orange', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), loss_f[adam_epochs:],
                    'g--', alpha=0.8, linewidth=1.5)
    
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', linewidth=2, label=r'$\mathcal{L}_{BC/IC}$')
    ax.plot([], [], 'orange', linewidth=2, label=r'$\mathcal{L}_{data}$')
    ax.plot([], [], 'g-', linewidth=2, label=r'$\mathcal{L}_{PDE}$')
    
    ax.set_ylim(ylim)
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel(ylabel, fontsize=FONT_CONFIG['axis_label'])
    ax.set_title(title, fontsize=FONT_CONFIG['title'])
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_CONFIG['legend'])
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close()


def _save_lambda_subplot(adam_epochs, total_iterations, lam_bc, lam_data, lam_f,
                         ylabel, title, ylim, filepath):
    """Save lambda weights subplot with consistent axis limits."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.semilogy(range(adam_epochs), lam_bc[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), lam_data[:adam_epochs], 'orange', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), lam_f[:adam_epochs], 'g-', alpha=0.8, linewidth=1.5)
    
    if len(lam_bc) > adam_epochs:
        ax.semilogy(range(adam_epochs, total_iterations), lam_bc[adam_epochs:],
                    'b--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), lam_data[adam_epochs:],
                    'orange', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), lam_f[adam_epochs:],
                    'g--', alpha=0.8, linewidth=1.5)
    
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', linewidth=2, label=r'$\lambda_{BC/IC}$')
    ax.plot([], [], 'orange', linewidth=2, label=r'$\lambda_{data}$')
    ax.plot([], [], 'g-', linewidth=2, label=r'$\lambda_{PDE}$')
    
    ax.set_ylim(ylim)
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel(ylabel, fontsize=FONT_CONFIG['axis_label'])
    ax.set_title(title, fontsize=FONT_CONFIG['title'])
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_CONFIG['legend'])
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close()


# =============================================================================
# EXPERIMENTAL DATA VISUALIZATION
# =============================================================================

def plot_experimental_comparison(u_pred, intensity_measured, x, y, t,
                                  diff_coeff_learned, output_dir='outputs',
                                  filename=None, roi_bounds=None,
                                  X_obs=None, save_subfigures=True):
    """
    Plot comparison for experimental data (intensity-based, no ground truth D).
    
    Similar to plot_2d_solution_comparison but tailored for experimental data:
    - Uses "Measured" instead of "True"
    - Shows target D range instead of single true value
    - Optionally overlays ROI boundary and observation points
    
    Args:
        u_pred: Predicted intensity
        intensity_measured: Measured intensity from experiment
        x, y, t: Coordinate arrays
        diff_coeff_learned: Learned D (normalized units)
        output_dir: Directory to save figures
        filename: Output filename (if None, generates unique name)
        roi_bounds: Optional dict with 'x_min', 'x_max', 'y_min', 'y_max' for ROI overlay
        X_obs: Observation points (N, 3) for overlay
        save_subfigures: Whether to save individual subfigures
    
    Returns:
        dict: Paths to saved files
    """
    return plot_2d_solution_comparison(
        u_pred=u_pred,
        usol=intensity_measured,
        x=x, y=y, t=t,
        diff_coeff_learned=diff_coeff_learned,
        diff_coeff_true=None,
        data_type='experimental',
        output_dir=output_dir,
        filename=filename,
        save_subfigures=save_subfigures
    )


def plot_training_diagnostics_experimental(history_adam, history_lbfgs,
                                            output_dir='outputs', filename=None,
                                            save_subfigures=True):
    """
    Plot training diagnostics for experimental data (no ground truth D).
    
    Wrapper that calls plot_training_diagnostics with data_type='experimental'.
    """
    return plot_training_diagnostics(
        history_adam=history_adam,
        history_lbfgs=history_lbfgs,
        diff_coeff_true=None,
        data_type='experimental',
        output_dir=output_dir,
        filename=filename,
        save_subfigures=save_subfigures
    )


# =============================================================================
# LATEX HELPER FUNCTIONS (Legacy - kept for backward compatibility)
# =============================================================================

def generate_latex_snippet(subfigure_paths, caption='', label='fig:results'):
    """
    Generate LaTeX code snippet for including subfigures in a tabular layout.
    
    NOTE: This is the legacy function. LaTeX snippets are now automatically
    generated and appended to latex_snippets.txt by the main plotting functions.
    
    Args:
        subfigure_paths: List of paths to subfigure PDFs
        caption: Figure caption text
        label: LaTeX label for the figure
    
    Returns:
        str: LaTeX code snippet
    """
    # Extract just filenames
    filenames = [os.path.basename(p) for p in subfigure_paths]
    
    # Determine number of columns (4 for standard layout)
    n_cols = min(4, len(filenames))
    
    snippet = r"""\checkoddpage
\begin{adjustwidth}{-1cm}{-1cm}
\centering
\setlength{\tabcolsep}{0pt}
\begin{tabular}{@{}""" + 'c@{}' * n_cols + r"""}
"""
    
    # Add column labels
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    snippet += '    ' + ' & '.join([f'\\multicolumn{{1}}{{c}}{{\\small {labels[i]}}}' 
                                    for i in range(min(n_cols, len(filenames)))]) + r' \\' + '\n'
    
    # Add images
    for row_start in range(0, len(filenames), n_cols):
        row_files = filenames[row_start:row_start + n_cols]
        img_commands = [f'\\includegraphics[width={0.95/n_cols:.2f}\\textwidth]{{Figs/{f}}}' 
                       for f in row_files]
        snippet += '    ' + ' &\n    '.join(img_commands) + r' \\' + '\n'
    
    snippet += r"""\end{tabular}
\end{adjustwidth}
\vspace{-0.1cm}
{\small \caption{""" + caption + r"""}}
\label{""" + label + r"""}
"""
    
    return snippet


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Legacy function signatures for backward compatibility
def plot_2d_solution_comparison_legacy(u_pred, usol, x, y, t, diff_coeff_learned,
                                        diff_coeff_true, output_dir='outputs',
                                        filename='diff2D_IDW_inverse.png'):
    """Legacy wrapper for backward compatibility."""
    return plot_2d_solution_comparison(
        u_pred=u_pred, usol=usol, x=x, y=y, t=t,
        diff_coeff_learned=diff_coeff_learned,
        diff_coeff_true=diff_coeff_true,
        data_type='numerical',
        output_dir=output_dir,
        filename=filename,
        save_subfigures=True
    )


def plot_training_diagnostics_legacy(history_adam, history_lbfgs, diff_coeff_true,
                                      output_dir='outputs',
                                      filename='inverse_diagnostics_2D.png'):
    """Legacy wrapper for backward compatibility."""
    return plot_training_diagnostics(
        history_adam=history_adam,
        history_lbfgs=history_lbfgs,
        diff_coeff_true=diff_coeff_true,
        data_type='numerical',
        output_dir=output_dir,
        filename=filename,
        save_subfigures=True
    )
"""
Visualization utilities for 2D diffusion inverse problems.

Provides publication-ready plots with:
- Large fonts/axes for Overleaf/presentations
- Unique identifiers (config params + datetime) in filenames
- Whole figures + separate subfigures for LaTeX integration
- Support for numerical (with true D=0.2) and experimental data (target D range)
- Percent error display instead of absolute error
- PDF output only (no PNG)
- Manual ylim parameters for consistent axis limits across runs

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
# AXIS LIMITS CONFIGURATION (Manual setting for cross-run consistency)
# These should match your YAML config: visualization.axis_limits
# =============================================================================

AXIS_LIMITS = {
    # Diagnostic plots
    'D_ylim': [1e-4, 1e0],           # Diffusion coefficient evolution
    'D_error_ylim': [1e-3, 1e2],     # D relative error (%)
    'loss_ylim': [1e-8, 1e2],        # Individual losses
    'total_loss_ylim': [1e-6, 1e1],  # Total loss
    'lambda_ylim': [1e-4, 1e2],      # IDW weights
    
    # Solution plots
    'u_vlim': [0.0, 1.0],            # Concentration field [vmin, vmax]
    'error_vlim': [0.0, 50.0],       # Percent error [vmin, vmax]
}


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

DPI_SAVE = 300

# Data type configurations
DATA_CONFIG = {
    'numerical': {
        'diff_coeff_true': 0.2,
        'diff_coeff_display': '0.2',
        'label': 'True D',
    },
    'experimental': {
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


def load_axis_limits_from_config(config):
    """
    Load axis limits from YAML config object.
    
    Expected YAML structure:
        visualization:
          axis_limits:
            D_ylim: [1e-4, 1e0]
            D_error_ylim: [1e-3, 1e2]
            loss_ylim: [1e-8, 1e2]
            total_loss_ylim: [1e-6, 1e1]
            lambda_ylim: [1e-4, 1e2]
            u_vlim: [0.0, 1.0]
            error_vlim: [0.0, 50.0]
    """
    global AXIS_LIMITS
    if hasattr(config, 'visualization') and hasattr(config.visualization, 'axis_limits'):
        limits = config.visualization.axis_limits
        for key in AXIS_LIMITS.keys():
            if hasattr(limits, key):
                AXIS_LIMITS[key] = getattr(limits, key)
        print(f"Loaded axis limits from config: {AXIS_LIMITS}")


# =============================================================================
# FILENAME GENERATION
# =============================================================================

def generate_unique_filename(base_name: str, extension: str = 'pdf', 
                             diff_coeff: float = None) -> str:
    """
    Generate unique filename with timestamp, UUID, and optionally D value.
    
    Format: {base_name}_D_{diff_coeff}_{timestamp}_{uuid}.{extension}
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    short_uuid = str(uuid.uuid4())[:8]
    
    if diff_coeff is not None:
        return f"{base_name}_D_{diff_coeff:.6f}_{timestamp}_{short_uuid}.{extension}"
    else:
        return f"{base_name}_{timestamp}_{short_uuid}.{extension}"


def _ensure_dirs(output_dir):
    """Ensure output and subfigures directories exist."""
    os.makedirs(output_dir, exist_ok=True)
    subfig_dir = os.path.join(output_dir, 'subfigures')
    os.makedirs(subfig_dir, exist_ok=True)
    return subfig_dir


# =============================================================================
# LATEX HELPER - Generate .txt file for Overleaf integration
# =============================================================================

def generate_latex_snippet(subfigure_paths: list, output_dir: str, 
                           base_name: str = 'latex_snippet') -> str:
    """
    Generate a .txt file with LaTeX code for including subfigures in Overleaf.
    
    Args:
        subfigure_paths: List of paths to subfigure PDF files
        output_dir: Directory to save the .txt file
        base_name: Base name for the .txt file
        
    Returns:
        Path to the generated .txt file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    txt_filename = f"{base_name}_{timestamp}.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    
    # Extract just filenames for LaTeX
    filenames = [os.path.basename(p) for p in subfigure_paths]
    
    latex_content = f"""% LaTeX snippet for including figures
% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
% Copy the subfigures/*.pdf files to your Overleaf Figs/ folder
% Then use this snippet in your document

% --- SOLUTION COMPARISON FIGURES ---
% Subfigures: {', '.join(filenames[:12] if len(filenames) > 12 else filenames)}

\\begin{{figure}}[htbp]
\\checkoddpage
\\begin{{adjustwidth}}{{-1cm}}{{-1cm}}
\\centering
\\setlength{{\\tabcolsep}}{{0pt}}
"""
    
    # Group by rows (4 per row for solution comparison)
    # Row labels: True/Measured, Predicted, % Error
    row_labels = ['True/Measured', 'Predicted', '% Error']
    
    for row_idx in range(min(3, (len(filenames) + 3) // 4)):
        start_idx = row_idx * 4
        end_idx = min(start_idx + 4, len(filenames))
        row_files = filenames[start_idx:end_idx]
        
        if row_idx == 0:
            # Add column headers
            latex_content += "\\begin{tabular}{@{}c@{}c@{}c@{}c@{}}\n"
            latex_content += "\\multicolumn{1}{c}{\\small (a)} & "
            latex_content += "\\multicolumn{1}{c}{\\small (b)} & "
            latex_content += "\\multicolumn{1}{c}{\\small (c)} & "
            latex_content += "\\multicolumn{1}{c}{\\small (d)} \\\\\n"
        
        # Add images for this row
        for i, fname in enumerate(row_files):
            width = "0.24\\textwidth"
            latex_content += f"\\includegraphics[width={width}]{{Figs/subfigures/{fname}}}"
            if i < len(row_files) - 1:
                latex_content += " & "
        
        latex_content += " \\\\\n"
    
    latex_content += """\\end{tabular}
\\end{adjustwidth}
\\vspace{-0.1cm}
{\\small \\caption{Solution comparison: Row 1 - True/Measured data at selected times; 
Row 2 - PINN predictions; Row 3 - Percent error between true and predicted.}}
\\label{fig:solution-comparison}
\\end{figure}

% --- DIAGNOSTICS FIGURES ---
% Use individual subfigures for custom arrangements

% Example single figure:
% \\begin{figure}[htbp]
% \\centering
% \\includegraphics[width=0.8\\textwidth]{Figs/subfigures/D_evolution.pdf}
% \\caption{Diffusion coefficient evolution during training.}
% \\label{fig:D-evolution}
% \\end{figure}

% Example 2x3 grid:
% \\begin{figure}[htbp]
% \\centering
% \\begin{tabular}{ccc}
% \\includegraphics[width=0.32\\textwidth]{Figs/subfigures/losses.pdf} &
% \\includegraphics[width=0.32\\textwidth]{Figs/subfigures/total_loss.pdf} &
% \\includegraphics[width=0.32\\textwidth]{Figs/subfigures/D_evolution.pdf} \\\\
% \\includegraphics[width=0.32\\textwidth]{Figs/subfigures/D_error.pdf} &
% \\includegraphics[width=0.32\\textwidth]{Figs/subfigures/lambdas_linear.pdf} &
% \\includegraphics[width=0.32\\textwidth]{Figs/subfigures/lambdas_log.pdf}
% \\end{tabular}
% \\caption{Training diagnostics.}
% \\label{fig:diagnostics}
% \\end{figure}
"""
    
    with open(txt_path, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX snippet saved: {txt_path}")
    return txt_path


# =============================================================================
# SOLUTION COMPARISON PLOT
# =============================================================================

def plot_2d_solution_comparison(u_pred, usol, x, y, t, diff_coeff_learned,
                                 diff_coeff_true=None, data_type=None,
                                 output_dir='outputs', filename=None,
                                 save_subfigures=True,
                                 u_vlim=None, error_vlim=None):
    """
    Plot 2D solution comparison with percent error.
    
    Args:
        u_pred: Predicted solution
        usol: True/measured solution
        x, y, t: Coordinate arrays
        diff_coeff_learned: Learned D value
        diff_coeff_true: True D (numerical) or None (experimental)
        data_type: 'numerical' or 'experimental' (defaults to DATA_TYPE)
        output_dir: Output directory
        filename: Custom filename (auto-generated if None)
        save_subfigures: Whether to save individual panels
        u_vlim: [vmin, vmax] for solution colorbar (uses AXIS_LIMITS if None)
        error_vlim: [vmin, vmax] for error colorbar (uses AXIS_LIMITS if None)
        
    Returns:
        dict: Paths to saved files
    """
    if data_type is None:
        data_type = DATA_TYPE
    
    # Use manual limits or fall back to config
    if u_vlim is None:
        u_vlim = AXIS_LIMITS['u_vlim']
    if error_vlim is None:
        error_vlim = AXIS_LIMITS['error_vlim']
    
    u_vmin, u_vmax = u_vlim
    err_vmin, err_vmax = error_vlim
    
    set_publication_style()
    subfig_dir = _ensure_dirs(output_dir)
    
    if filename is None:
        filename = generate_unique_filename('solution_comparison', 'pdf', diff_coeff_learned)
    
    # Reshape prediction if needed
    if u_pred.ndim == 2 and u_pred.shape[1] == 1:
        u_pred = u_pred.flatten()
    if u_pred.ndim == 1:
        u_pred_reshaped = np.reshape(u_pred, usol.shape, order='C')
    else:
        u_pred_reshaped = u_pred
    
    # Select time indices
    nt = len(t)
    t_indices = [0, nt//3, 2*nt//3, nt-1]
    n_cols = len(t_indices)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Compute percent error
    usol_abs_max = np.maximum(np.abs(usol), 1e-10)
    percent_error_all = 100.0 * np.abs(usol - u_pred_reshaped) / usol_abs_max
    percent_error_all = np.clip(percent_error_all, 0, 100)
    
    # Labels based on data type
    if data_type == 'numerical':
        true_label = 'True'
        d_display = f'True D={DATA_CONFIG["numerical"]["diff_coeff_true"]}'
    else:
        true_label = 'Measured'
        d_range = DATA_CONFIG['experimental']['diff_coeff_range']
        d_display = f'Target D: {d_range[0]:.4f}-{d_range[1]:.4f}'
    
    # Create figure
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
        
        if save_subfigures:
            subfig_path = os.path.join(subfig_dir, 
                f'{true_label.lower()}_at_T{i}_D_{diff_coeff_learned:.6f}.pdf')
            _save_single_panel(X, Y, usol[:, :, ti], 'jet', u_vmin, u_vmax,
                              f'{true_label}, t={t[ti]:.3f}s', 'Concentration', subfig_path)
            saved_subfigures.append(subfig_path)
    
    cax0 = fig.add_subplot(gs[0, n_cols])
    cbar0 = fig.colorbar(im0, cax=cax0)
    cbar0.set_label('Concentration', fontsize=FONT_CONFIG['colorbar_label'])
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
        
        if save_subfigures:
            subfig_path = os.path.join(subfig_dir,
                f'pred_at_T{i}_D_{diff_coeff_learned:.6f}.pdf')
            _save_single_panel(X, Y, u_pred_reshaped[:, :, ti], 'jet', u_vmin, u_vmax,
                              f'Predicted, t={t[ti]:.3f}s', 'Concentration', subfig_path)
            saved_subfigures.append(subfig_path)
    
    cax1 = fig.add_subplot(gs[1, n_cols])
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('Concentration', fontsize=FONT_CONFIG['colorbar_label'])
    cbar1.ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Row 2: Percent error
    axes_err = [fig.add_subplot(gs[2, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        im2 = axes_err[i].pcolormesh(X, Y, percent_error_all[:, :, ti], cmap='hot',
                                      shading='auto', vmin=err_vmin, vmax=err_vmax)
        axes_err[i].set_title(f'% Error, t={t[ti]:.3f}s', fontsize=FONT_CONFIG['title'])
        axes_err[i].set_xlabel('x', fontsize=FONT_CONFIG['axis_label'])
        if i == 0:
            axes_err[i].set_ylabel('y', fontsize=FONT_CONFIG['axis_label'])
        axes_err[i].set_aspect('equal')
        axes_err[i].tick_params(labelsize=FONT_CONFIG['tick_label'])
        
        if save_subfigures:
            subfig_path = os.path.join(subfig_dir,
                f'percent_error_at_T{i}_D_{diff_coeff_learned:.6f}.pdf')
            _save_single_panel(X, Y, percent_error_all[:, :, ti], 'hot', err_vmin, err_vmax,
                              f'% Error, t={t[ti]:.3f}s', '% Error', subfig_path)
            saved_subfigures.append(subfig_path)
    
    cax2 = fig.add_subplot(gs[2, n_cols])
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('% Error', fontsize=FONT_CONFIG['colorbar_label'])
    cbar2.ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    plt.suptitle(f'2D Diffusion: Learned D = {diff_coeff_learned:.6f} ({d_display})',
                 fontsize=FONT_CONFIG['suptitle'])
    
    # Save PDF only
    filepath = os.path.join(output_dir, filename)
    if not filepath.endswith('.pdf'):
        filepath = filepath.replace('.png', '.pdf')
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()
    
    # Generate LaTeX snippet
    if save_subfigures and saved_subfigures:
        latex_path = generate_latex_snippet(saved_subfigures, output_dir, 'solution_latex')
    
    return {'whole': filepath, 'subfigures': saved_subfigures}


def _save_single_panel(X, Y, data, cmap, vmin, vmax, title, cbar_label, filepath):
    """Save a single panel as a standalone PDF figure for LaTeX."""
    fig_sub, ax_sub = plt.subplots(figsize=(5, 4))
    im = ax_sub.pcolormesh(X, Y, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    ax_sub.set_title(title, fontsize=FONT_CONFIG['title'])
    ax_sub.set_xlabel('x', fontsize=FONT_CONFIG['axis_label'])
    ax_sub.set_ylabel('y', fontsize=FONT_CONFIG['axis_label'])
    ax_sub.set_aspect('equal')
    ax_sub.tick_params(labelsize=FONT_CONFIG['tick_label'])
    cbar = fig_sub.colorbar(im, ax=ax_sub)
    cbar.set_label(cbar_label, fontsize=FONT_CONFIG['colorbar_label'])
    cbar.ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close(fig_sub)
    print(f"  Subfigure: {filepath}")


# =============================================================================
# TRAINING DIAGNOSTICS PLOT
# =============================================================================

def plot_training_diagnostics(history_adam, history_lbfgs, diff_coeff_true=None,
                               data_type=None, output_dir='outputs',
                               filename=None, save_subfigures=True,
                               D_ylim=None, D_error_ylim=None, loss_ylim=None,
                               total_loss_ylim=None, lambda_ylim=None):
    """
    Plot training diagnostics with 6 panels.
    
    Args:
        history_adam: Dict with Adam training history
        history_lbfgs: Dict with L-BFGS training history
        diff_coeff_true: True D value (numerical) or None
        data_type: 'numerical' or 'experimental'
        output_dir: Output directory
        filename: Custom filename
        save_subfigures: Save individual panels
        D_ylim: [ymin, ymax] for D evolution plot
        D_error_ylim: [ymin, ymax] for D error plot
        loss_ylim: [ymin, ymax] for individual losses
        total_loss_ylim: [ymin, ymax] for total loss
        lambda_ylim: [ymin, ymax] for lambda weights
        
    Returns:
        dict: Paths to saved files
    """
    if data_type is None:
        data_type = DATA_TYPE
    
    # Use manual limits or config defaults
    if D_ylim is None:
        D_ylim = AXIS_LIMITS['D_ylim']
    if D_error_ylim is None:
        D_error_ylim = AXIS_LIMITS['D_error_ylim']
    if loss_ylim is None:
        loss_ylim = AXIS_LIMITS['loss_ylim']
    if total_loss_ylim is None:
        total_loss_ylim = AXIS_LIMITS['total_loss_ylim']
    if lambda_ylim is None:
        lambda_ylim = AXIS_LIMITS['lambda_ylim']
    
    set_publication_style()
    subfig_dir = _ensure_dirs(output_dir)
    
    # Get final D for filename
    final_D = history_lbfgs.get('diff_coeff', history_adam.get('diff_coeff', [0.5]))[-1]
    if filename is None:
        filename = generate_unique_filename('training_diagnostics', 'pdf', final_D)
    
    # Combine histories
    adam_epochs = len(history_adam.get('total_loss', []))
    
    diff_coeff_all = np.concatenate([
        history_adam.get('diff_coeff', []),
        history_lbfgs.get('diff_coeff', [])
    ])
    total_loss_all = np.concatenate([
        history_adam.get('total_loss', []),
        history_lbfgs.get('total_loss', [])
    ])
    loss_bc_all = np.concatenate([
        history_adam.get('loss_bc', []),
        history_lbfgs.get('loss_bc', [])
    ])
    loss_data_all = np.concatenate([
        history_adam.get('loss_data', []),
        history_lbfgs.get('loss_data', [])
    ])
    loss_f_all = np.concatenate([
        history_adam.get('loss_f', []),
        history_lbfgs.get('loss_f', [])
    ])
    lam_bc_all = np.concatenate([
        history_adam.get('lam_bc', []),
        history_lbfgs.get('lam_bc', [])
    ])
    lam_data_all = np.concatenate([
        history_adam.get('lam_data', []),
        history_lbfgs.get('lam_data', [])
    ])
    lam_f_all = np.concatenate([
        history_adam.get('lam_f', []),
        history_lbfgs.get('lam_f', [])
    ])
    
    total_iterations = len(total_loss_all)
    
    # Compute D error based on data type
    if data_type == 'numerical' and diff_coeff_true is not None:
        d_error_all = 100.0 * np.abs(diff_coeff_all - diff_coeff_true) / diff_coeff_true
        d_ref_value = diff_coeff_true
        d_label = f'True D = {diff_coeff_true}'
    else:
        # For experimental: compute error relative to target range midpoint
        d_range = DATA_CONFIG['experimental']['diff_coeff_range']
        d_mid = (d_range[0] + d_range[1]) / 2
        d_error_all = 100.0 * np.abs(diff_coeff_all - d_mid) / d_mid
        d_ref_value = None
        d_label = f'Target: {d_range[0]:.4f}-{d_range[1]:.4f}'
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    saved_subfigures = []
    
    # Plot 1: Individual losses (log)
    ax = axes[0, 0]
    ax.semilogy(range(adam_epochs), loss_bc_all[:adam_epochs], 'b-', 
                alpha=0.8, linewidth=1.5, label=r'$\mathcal{L}_{BC/IC}$')
    ax.semilogy(range(adam_epochs), loss_data_all[:adam_epochs], 'orange', 
                alpha=0.8, linewidth=1.5, label=r'$\mathcal{L}_{data}$')
    ax.semilogy(range(adam_epochs), loss_f_all[:adam_epochs], 'g-', 
                alpha=0.8, linewidth=1.5, label=r'$\mathcal{L}_{PDE}$')
    if history_lbfgs.get('loss_bc'):
        ax.semilogy(range(adam_epochs, total_iterations), loss_bc_all[adam_epochs:],
                    'b--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), loss_data_all[adam_epochs:],
                    'orange', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), loss_f_all[adam_epochs:],
                    'g--', alpha=0.8, linewidth=1.5)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Loss', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('Individual Losses', fontsize=FONT_CONFIG['title'])
    ax.set_ylim(loss_ylim)
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        _save_diagnostic_subplot(ax, os.path.join(subfig_dir, 'losses.pdf'),
                                 ylim=loss_ylim, yscale='log')
        saved_subfigures.append(os.path.join(subfig_dir, 'losses.pdf'))
    
    # Plot 2: Total loss (log)
    ax = axes[0, 1]
    ax.semilogy(range(adam_epochs), total_loss_all[:adam_epochs], 'b-', 
                linewidth=2, label='Adam')
    if history_lbfgs.get('total_loss'):
        ax.semilogy(range(adam_epochs, total_iterations), total_loss_all[adam_epochs:],
                    'g-', linewidth=2, label='L-BFGS')
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Total Loss', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('Total Loss Evolution', fontsize=FONT_CONFIG['title'])
    ax.set_ylim(total_loss_ylim)
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        _save_diagnostic_subplot(ax, os.path.join(subfig_dir, 'total_loss.pdf'),
                                 ylim=total_loss_ylim, yscale='log')
        saved_subfigures.append(os.path.join(subfig_dir, 'total_loss.pdf'))
    
    # Plot 3: D evolution (log)
    ax = axes[0, 2]
    ax.semilogy(range(adam_epochs), diff_coeff_all[:adam_epochs], 'b-', 
                linewidth=2, label='Adam')
    if history_lbfgs.get('diff_coeff'):
        ax.semilogy(range(adam_epochs, total_iterations), diff_coeff_all[adam_epochs:],
                    'g-', linewidth=2, label='L-BFGS')
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Add reference line(s)
    if data_type == 'numerical' and d_ref_value is not None:
        ax.axhline(y=d_ref_value, color='r', linestyle='--', linewidth=1.5, label=d_label)
    else:
        d_range = DATA_CONFIG['experimental']['diff_coeff_range']
        ax.axhspan(d_range[0], d_range[1], alpha=0.3, color='green', label=d_label)
    
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Diffusion Coefficient', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('D Evolution (Log Scale)', fontsize=FONT_CONFIG['title'])
    ax.set_ylim(D_ylim)
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        _save_diagnostic_subplot(ax, os.path.join(subfig_dir, 'D_evolution.pdf'),
                                 ylim=D_ylim, yscale='log')
        saved_subfigures.append(os.path.join(subfig_dir, 'D_evolution.pdf'))
    
    # Plot 4: D error (log)
    ax = axes[1, 0]
    ax.semilogy(range(adam_epochs), d_error_all[:adam_epochs], 'b-', 
                linewidth=2, label='Adam')
    if history_lbfgs.get('diff_coeff'):
        ax.semilogy(range(adam_epochs, total_iterations), d_error_all[adam_epochs:],
                    'g-', linewidth=2, label='L-BFGS')
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Relative Error (%)', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('D Error (Log Scale)', fontsize=FONT_CONFIG['title'])
    ax.set_ylim(D_error_ylim)
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        _save_diagnostic_subplot(ax, os.path.join(subfig_dir, 'D_error.pdf'),
                                 ylim=D_error_ylim, yscale='log')
        saved_subfigures.append(os.path.join(subfig_dir, 'D_error.pdf'))
    
    # Plot 5: Lambda weights (linear)
    ax = axes[1, 1]
    ax.plot(range(adam_epochs), lam_bc_all[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
    ax.plot(range(adam_epochs), lam_data_all[:adam_epochs], 'orange', alpha=0.8, linewidth=1.5)
    ax.plot(range(adam_epochs), lam_f_all[:adam_epochs], 'g-', alpha=0.8, linewidth=1.5)
    if history_lbfgs.get('lam_bc'):
        ax.plot(range(adam_epochs, total_iterations), lam_bc_all[adam_epochs:],
                'b--', alpha=0.8, linewidth=1.5)
        ax.plot(range(adam_epochs, total_iterations), lam_data_all[adam_epochs:],
                'orange', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.plot(range(adam_epochs, total_iterations), lam_f_all[adam_epochs:],
                'g--', alpha=0.8, linewidth=1.5)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', linewidth=2, label=r'$\lambda_{BC/IC}$')
    ax.plot([], [], 'orange', linewidth=2, label=r'$\lambda_{data}$')
    ax.plot([], [], 'g-', linewidth=2, label=r'$\lambda_{PDE}$')
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Weight', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('IDW Weights (Linear Scale)', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        _save_diagnostic_subplot(ax, os.path.join(subfig_dir, 'lambdas_linear.pdf'),
                                 ylim=None, yscale='linear')
        saved_subfigures.append(os.path.join(subfig_dir, 'lambdas_linear.pdf'))
    
    # Plot 6: Lambda weights (log)
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
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Weight (log)', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('IDW Weights (Log Scale)', fontsize=FONT_CONFIG['title'])
    ax.set_ylim(lambda_ylim)
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        _save_diagnostic_subplot(ax, os.path.join(subfig_dir, 'lambdas_log.pdf'),
                                 ylim=lambda_ylim, yscale='log')
        saved_subfigures.append(os.path.join(subfig_dir, 'lambdas_log.pdf'))
    
    plt.tight_layout()
    
    # Save PDF only
    filepath = os.path.join(output_dir, filename)
    if not filepath.endswith('.pdf'):
        filepath = filepath.replace('.png', '.pdf')
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()
    
    # Generate LaTeX snippet
    if save_subfigures and saved_subfigures:
        latex_path = generate_latex_snippet(saved_subfigures, output_dir, 'diagnostics_latex')
    
    return {'whole': filepath, 'subfigures': saved_subfigures}


def _save_diagnostic_subplot(ax, filepath, ylim=None, yscale='log'):
    """Save an individual diagnostic subplot as a standalone PDF figure."""
    fig_sub, ax_sub = plt.subplots(figsize=(6, 5))
    
    # Copy all lines
    for line in ax.get_lines():
        ax_sub.plot(line.get_xdata(), line.get_ydata(),
                   linestyle=line.get_linestyle(),
                   color=line.get_color(),
                   linewidth=line.get_linewidth(),
                   alpha=line.get_alpha(),
                   label=line.get_label())
    
    # Set scale and limits
    ax_sub.set_xscale(ax.get_xscale())
    ax_sub.set_yscale(yscale)
    if ylim is not None:
        ax_sub.set_ylim(ylim)
    
    # Copy labels and title
    ax_sub.set_xlabel(ax.get_xlabel(), fontsize=FONT_CONFIG['axis_label'])
    ax_sub.set_ylabel(ax.get_ylabel(), fontsize=FONT_CONFIG['axis_label'])
    ax_sub.set_title(ax.get_title(), fontsize=FONT_CONFIG['title'])
    ax_sub.tick_params(labelsize=FONT_CONFIG['tick_label'])
    ax_sub.grid(True, alpha=0.3)
    
    # Only add legend if there are labeled items
    handles, labels = ax_sub.get_legend_handles_labels()
    if any(label and not label.startswith('_') for label in labels):
        ax_sub.legend(fontsize=FONT_CONFIG['legend'])
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close(fig_sub)
    print(f"  Subfigure: {filepath}")


# =============================================================================
# EXPERIMENTAL DATA VISUALIZATION
# =============================================================================

def plot_experimental_comparison(u_pred, intensity_measured, x, y, t,
                                  diff_coeff_learned, output_dir='outputs',
                                  filename=None, roi_bounds=None,
                                  X_obs=None, save_subfigures=True,
                                  u_vlim=None, error_vlim=None):
    """
    Plot comparison for experimental data (intensity-based, no ground truth D).
    
    This is a wrapper around plot_2d_solution_comparison for experimental data.
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
        save_subfigures=save_subfigures,
        u_vlim=u_vlim,
        error_vlim=error_vlim
    )
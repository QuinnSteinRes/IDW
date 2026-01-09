"""
Visualization utilities for 2D diffusion inverse problems.
Updated for LaTeX/Overleaf and poster compatibility with larger fonts.

Provides:
- Solution comparison plots (true vs predicted vs error at multiple time slices)
- Training diagnostics (parameter evolution, losses, IDW weights)
- Unique file identifiers to prevent overwriting
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import uuid
from datetime import datetime


# =============================================================================
# Font configuration for LaTeX/poster readability
# =============================================================================
FONT_CONFIG = {
    'title': 16,
    'suptitle': 18,
    'axis_label': 14,
    'tick_label': 12,
    'colorbar_label': 13,
    'legend': 12,
}


def generate_unique_filename(base_name: str, extension: str = 'png') -> str:
    """
    Generate a unique filename with timestamp and short UUID.
    
    Args:
        base_name: Base name for the file (e.g., 'diff2D_IDW_inverse')
        extension: File extension (default: 'png')
    
    Returns:
        Unique filename string, e.g., 'diff2D_IDW_inverse_090126_'
    """
    timestamp = datetime.now().strftime('%d%m%y_%H%M%S')
    short_uuid = uuid.uuid4().hex[:3]
    return f"{base_name}_{timestamp}_{short_uuid}.{extension}"


def plot_2d_solution_comparison(u_pred, usol, x, y, t, diff_coeff_learned, 
                                diff_coeff_true, output_dir='outputs',
                                filename=None):
    """
    Plot 2D solution comparison at 4 time slices.
    
    Creates a 3-row figure:
    - Row 0: True solution at 4 time points
    - Row 1: Predicted solution at 4 time points  
    - Row 2: Absolute error at 4 time points
    
    Args:
        u_pred: Predicted solution, shape (Nx+1, Ny+1, Nt) or flattened
        usol: True solution, shape (Nx+1, Ny+1, Nt)
        x: x coordinates (Nx+1,)
        y: y coordinates (Ny+1,)
        t: time coordinates (Nt,)
        diff_coeff_learned: Learned diffusion coefficient
        diff_coeff_true: True diffusion coefficient
        output_dir: Directory to save figure
        filename: Output filename (if None, generates unique name)
    
    Returns:
        filepath: Path to saved figure
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename if not provided
    if filename is None:
        filename = generate_unique_filename('diff2D_IDW_inverse')
    
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
    
    # Compute global colorbar limits
    u_vmin = min(usol.min(), u_pred_reshaped.min())
    u_vmax = max(usol.max(), u_pred_reshaped.max())
    
    error_all = np.abs(usol - u_pred_reshaped)
    err_vmin = 0
    err_vmax = error_all.max()
    
    # Create figure with GridSpec (increased size for readability)
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, n_cols + 1, width_ratios=[1, 1, 1, 1, 0.05], 
                           wspace=0.3, hspace=0.35)
    
    # Row 0: True solution
    axes_true = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        im0 = axes_true[i].pcolormesh(X, Y, usol[:, :, ti], cmap='jet', shading='auto',
                                       vmin=u_vmin, vmax=u_vmax)
        axes_true[i].set_title(f'True, t={t[ti]:.3f}s', fontsize=FONT_CONFIG['title'])
        axes_true[i].set_xlabel('x', fontsize=FONT_CONFIG['axis_label'])
        if i == 0:
            axes_true[i].set_ylabel('y', fontsize=FONT_CONFIG['axis_label'])
        axes_true[i].set_aspect('equal')
        axes_true[i].tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Colorbar for row 0
    cax0 = fig.add_subplot(gs[0, n_cols])
    cbar0 = fig.colorbar(im0, cax=cax0)
    cbar0.set_label('u (True)', fontsize=FONT_CONFIG['colorbar_label'])
    cbar0.ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Row 1: Predicted solution
    axes_pred = [fig.add_subplot(gs[1, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        im1 = axes_pred[i].pcolormesh(X, Y, u_pred_reshaped[:, :, ti], cmap='jet', shading='auto',
                                       vmin=u_vmin, vmax=u_vmax)
        axes_pred[i].set_title(f'Predicted, t={t[ti]:.3f}s', fontsize=FONT_CONFIG['title'])
        axes_pred[i].set_xlabel('x', fontsize=FONT_CONFIG['axis_label'])
        if i == 0:
            axes_pred[i].set_ylabel('y', fontsize=FONT_CONFIG['axis_label'])
        axes_pred[i].set_aspect('equal')
        axes_pred[i].tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Colorbar for row 1
    cax1 = fig.add_subplot(gs[1, n_cols])
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('u (Pred)', fontsize=FONT_CONFIG['colorbar_label'])
    cbar1.ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Row 2: Error
    axes_err = [fig.add_subplot(gs[2, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        error = np.abs(usol[:, :, ti] - u_pred_reshaped[:, :, ti])
        im2 = axes_err[i].pcolormesh(X, Y, error, cmap='hot', shading='auto',
                                      vmin=err_vmin, vmax=err_vmax)
        axes_err[i].set_title(f'|Error|, t={t[ti]:.3f}s', fontsize=FONT_CONFIG['title'])
        axes_err[i].set_xlabel('x', fontsize=FONT_CONFIG['axis_label'])
        if i == 0:
            axes_err[i].set_ylabel('y', fontsize=FONT_CONFIG['axis_label'])
        axes_err[i].set_aspect('equal')
        axes_err[i].tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Colorbar for row 2
    cax2 = fig.add_subplot(gs[2, n_cols])
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('|Error|', fontsize=FONT_CONFIG['colorbar_label'])
    cbar2.ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    plt.suptitle(f'2D Diffusion: Learned D = {diff_coeff_learned:.4f} (True: {diff_coeff_true})', 
                 fontsize=FONT_CONFIG['suptitle'])
    
    # Save figure
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()
    
    return filepath


def plot_training_diagnostics(history_adam, history_lbfgs, diff_coeff_true,
                               output_dir='outputs',
                               filename=None):
    """
    Plot comprehensive training diagnostics in 2x3 layout.
    
    Creates diagnostic plots showing:
    - D evolution (linear and log scale)
    - D error evolution (log scale)
    - Loss components (BC, Data, PDE) over epochs
    - Lambda weights (linear and log scale)
    
    Combines Adam and L-BFGS histories, marking the phase transition.
    
    Args:
        history_adam: Dict with keys 'diff_coeff', 'loss_bc', 'loss_data', 'loss_f',
                      'lam_bc', 'lam_data', 'lam_f' from Adam phase
        history_lbfgs: Dict with same keys from L-BFGS phase
        diff_coeff_true: True diffusion coefficient
        output_dir: Directory to save figure
        filename: Output filename (if None, generates unique name)
    
    Returns:
        filepath: Path to saved figure
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename if not provided
    if filename is None:
        filename = generate_unique_filename('inverse_diagnostics_2D')
    
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
    
    # Create 2x3 figure (increased size)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: D evolution (linear)
    ax = axes[0, 0]
    ax.plot(range(adam_epochs), history_adam['diff_coeff'], 'b-', alpha=0.8, linewidth=1.5)
    if history_lbfgs.get('diff_coeff'):
        ax.plot(range(adam_epochs, total_iterations), history_lbfgs['diff_coeff'], 
                'b--', alpha=0.8, linewidth=1.5)
    ax.axhline(y=diff_coeff_true, color='r', linestyle='-', linewidth=2, label=f'True D={diff_coeff_true}')
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('D', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('Diffusion Coefficient Evolution', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Plot 2: D evolution (log scale)
    ax = axes[0, 1]
    ax.semilogy(range(adam_epochs), history_adam['diff_coeff'], 'b-', alpha=0.8, linewidth=1.5)
    if history_lbfgs.get('diff_coeff'):
        ax.semilogy(range(adam_epochs, total_iterations), history_lbfgs['diff_coeff'], 
                    'b--', alpha=0.8, linewidth=1.5)
    ax.axhline(y=diff_coeff_true, color='r', linestyle='-', linewidth=2, label=f'True D={diff_coeff_true}')
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('D (log)', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('D Evolution (Log Scale)', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Plot 3: D error evolution
    ax = axes[0, 2]
    d_error = np.abs(np.array(diff_all) - diff_coeff_true)
    ax.semilogy(range(adam_epochs), d_error[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
    if len(d_error) > adam_epochs:
        ax.semilogy(range(adam_epochs, total_iterations), d_error[adam_epochs:], 
                    'b--', alpha=0.8, linewidth=1.5)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('|D - D_true|', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('D Error Evolution', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Plot 4: Loss components
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
    ax.plot([], [], 'b-', linewidth=2, label=r'$\mathcal{L}_{BC}$')
    ax.plot([], [], 'orange', linewidth=2, label=r'$\mathcal{L}_{data}$')
    ax.plot([], [], 'g-', linewidth=2, label=r'$\mathcal{L}_{PDE}$')
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Loss', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('Loss Components', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
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
    ax.plot([], [], 'b-', linewidth=2, label=r'$\lambda_{BC}$')
    ax.plot([], [], 'orange', linewidth=2, label=r'$\lambda_{data}$')
    ax.plot([], [], 'g-', linewidth=2, label=r'$\lambda_{PDE}$')
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Weight', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('IDW Weights', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
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
    ax.plot([], [], 'b-', linewidth=2, label=r'$\lambda_{BC}$')
    ax.plot([], [], 'orange', linewidth=2, label=r'$\lambda_{data}$')
    ax.plot([], [], 'g-', linewidth=2, label=r'$\lambda_{PDE}$')
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Weight (log)', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('IDW Weights (Log Scale)', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()
    
    return filepath
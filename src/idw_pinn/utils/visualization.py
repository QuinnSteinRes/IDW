"""
Visualization utilities for 2D diffusion inverse problems.
Extracted from legacy: 2D_num_inv_IDW_newPrintOut_newFigs.py

Provides:
- Solution comparison plots (true vs predicted vs error at multiple time slices)
- Training diagnostics (parameter evolution, losses, IDW weights)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def plot_2d_solution_comparison(u_pred, usol, x, y, t, diff_coeff_learned, 
                                  diff_coeff_true, output_dir='outputs',
                                  filename='diff2D_IDW_inverse.png'):
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
        filename: Output filename
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Reshape prediction to match solution shape if needed
    if u_pred.ndim == 2 and u_pred.shape[1] == 1:
        # Flatten if it's (N, 1) shaped
        u_pred = u_pred.flatten()
    
    if u_pred.ndim == 1:
        # Reshape from flattened to (Nx+1, Ny+1, Nt)
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
    
    # Create figure with GridSpec (4 columns + 1 for colorbar)
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, n_cols + 1, width_ratios=[1, 1, 1, 1, 0.05], 
                           wspace=0.25, hspace=0.3)
    
    # Row 0: True solution
    axes_true = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        im0 = axes_true[i].pcolormesh(X, Y, usol[:, :, ti], cmap='jet', shading='auto',
                                       vmin=u_vmin, vmax=u_vmax)
        axes_true[i].set_title(f'True, t={t[ti]:.3f}s', fontsize=10)
        axes_true[i].set_xlabel('x', fontsize=9)
        if i == 0:
            axes_true[i].set_ylabel('y', fontsize=9)
        axes_true[i].set_aspect('equal')
        axes_true[i].tick_params(labelsize=8)
    
    # Colorbar for row 0
    cax0 = fig.add_subplot(gs[0, n_cols])
    cbar0 = fig.colorbar(im0, cax=cax0)
    cbar0.set_label('u (True)', fontsize=9)
    cbar0.ax.tick_params(labelsize=8)
    
    # Row 1: Predicted solution
    axes_pred = [fig.add_subplot(gs[1, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        im1 = axes_pred[i].pcolormesh(X, Y, u_pred_reshaped[:, :, ti], cmap='jet', shading='auto',
                                       vmin=u_vmin, vmax=u_vmax)
        axes_pred[i].set_title(f'Predicted, t={t[ti]:.3f}s', fontsize=10)
        axes_pred[i].set_xlabel('x', fontsize=9)
        if i == 0:
            axes_pred[i].set_ylabel('y', fontsize=9)
        axes_pred[i].set_aspect('equal')
        axes_pred[i].tick_params(labelsize=8)
    
    # Colorbar for row 1
    cax1 = fig.add_subplot(gs[1, n_cols])
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('u (Pred)', fontsize=9)
    cbar1.ax.tick_params(labelsize=8)
    
    # Row 2: Error
    axes_err = [fig.add_subplot(gs[2, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        error = np.abs(usol[:, :, ti] - u_pred_reshaped[:, :, ti])
        im2 = axes_err[i].pcolormesh(X, Y, error, cmap='hot', shading='auto',
                                      vmin=err_vmin, vmax=err_vmax)
        axes_err[i].set_title(f'|Error|, t={t[ti]:.3f}s', fontsize=10)
        axes_err[i].set_xlabel('x', fontsize=9)
        if i == 0:
            axes_err[i].set_ylabel('y', fontsize=9)
        axes_err[i].set_aspect('equal')
        axes_err[i].tick_params(labelsize=8)
    
    # Colorbar for row 2
    cax2 = fig.add_subplot(gs[2, n_cols])
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('|Error|', fontsize=9)
    cbar2.ax.tick_params(labelsize=8)
    
    plt.suptitle(f'2D Diffusion: Learned D = {diff_coeff_learned:.4f} (True: {diff_coeff_true})', 
                 fontsize=14)
    
    # Save figure
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_training_diagnostics(history_adam, history_lbfgs, diff_coeff_true,
                               output_dir='outputs',
                               filename='inverse_diagnostics_2D.png'):
    """
    Plot comprehensive training diagnostics in 2x3 layout.
    
    Creates diagnostic plots showing:
    - D evolution (linear and log scale)
    - D error evolution (log scale)
    - Loss components (BC, Data, PDE) over epochs
    - Lambda weights (linear and log scale)
    
    Combines Adam and L-BFGS histories, marking the phase transition.
    
    Args:
        history_adam: Dict with keys: diff_coeff, loss_bc, loss_data, loss_f, 
                      lam_bc, lam_data, lam_f (each a list from Adam phase)
        history_lbfgs: Dict with same keys (each a list from L-BFGS phase)
        diff_coeff_true: True diffusion coefficient value
        output_dir: Directory to save figure
        filename: Output filename
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine histories
    diff_coeff_all = history_adam['diff_coeff'] + history_lbfgs['diff_coeff']
    loss_bc_all = history_adam['loss_bc'] + history_lbfgs['loss_bc']
    loss_data_all = history_adam['loss_data'] + history_lbfgs['loss_data']
    loss_f_all = history_adam['loss_f'] + history_lbfgs['loss_f']
    lam_bc_all = history_adam['lam_bc'] + history_lbfgs['lam_bc']
    lam_data_all = history_adam['lam_data'] + history_lbfgs['lam_data']
    lam_f_all = history_adam['lam_f'] + history_lbfgs['lam_f']
    
    adam_epochs = len(history_adam['diff_coeff'])
    total_iterations = len(diff_coeff_all)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: D evolution (linear)
    ax = axes[0, 0]
    ax.plot(range(adam_epochs), history_adam['diff_coeff'], 'b-', label='Adam phase')
    if history_lbfgs['diff_coeff']:
        ax.plot(range(adam_epochs, total_iterations), history_lbfgs['diff_coeff'], 
                'g-', label='L-BFGS phase')
    ax.axhline(y=diff_coeff_true, color='r', linestyle='--', 
               label=f'True D = {diff_coeff_true}')
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Diffusion Coefficient')
    ax.set_title('D Evolution (Linear Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: D evolution (log)
    ax = axes[0, 1]
    ax.semilogy(range(adam_epochs), history_adam['diff_coeff'], 'b-', label='Adam phase')
    if history_lbfgs['diff_coeff']:
        ax.semilogy(range(adam_epochs, total_iterations), history_lbfgs['diff_coeff'], 
                    'g-', label='L-BFGS phase')
    ax.axhline(y=diff_coeff_true, color='r', linestyle='--', 
               label=f'True D = {diff_coeff_true}')
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Diffusion Coefficient (log)')
    ax.set_title('D Evolution (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: D error (log)
    ax = axes[0, 2]
    d_errors_adam = np.abs(np.array(history_adam['diff_coeff']) - diff_coeff_true)
    ax.semilogy(range(adam_epochs), d_errors_adam, 'b-', label='Adam phase')
    if history_lbfgs['diff_coeff']:
        d_errors_lbfgs = np.abs(np.array(history_lbfgs['diff_coeff']) - diff_coeff_true)
        ax.semilogy(range(adam_epochs, total_iterations), d_errors_lbfgs, 
                    'g-', label='L-BFGS phase')
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('|D - D_true| (log scale)')
    ax.set_title('Diffusion Coefficient Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Loss components
    ax = axes[1, 0]
    ax.semilogy(range(adam_epochs), loss_bc_all[:adam_epochs], 'b-', alpha=0.8)
    ax.semilogy(range(adam_epochs), loss_data_all[:adam_epochs], 'orange', alpha=0.8)
    ax.semilogy(range(adam_epochs), loss_f_all[:adam_epochs], 'g-', alpha=0.8)
    if history_lbfgs['loss_bc']:
        ax.semilogy(range(adam_epochs, total_iterations), loss_bc_all[adam_epochs:], 
                    'b--', alpha=0.8)
        ax.semilogy(range(adam_epochs, total_iterations), loss_data_all[adam_epochs:], 
                    'orange', linestyle='--', alpha=0.8)
        ax.semilogy(range(adam_epochs, total_iterations), loss_f_all[adam_epochs:], 
                    'g--', alpha=0.8)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', label='L_bc (BC/IC)')
    ax.plot([], [], 'orange', label='L_data (Interior)')
    ax.plot([], [], 'g-', label='L_f (PDE)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components (solid=Adam, dashed=L-BFGS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Lambda weights (linear)
    ax = axes[1, 1]
    ax.plot(range(adam_epochs), lam_bc_all[:adam_epochs], 'b-', alpha=0.8)
    ax.plot(range(adam_epochs), lam_data_all[:adam_epochs], 'orange', alpha=0.8)
    ax.plot(range(adam_epochs), lam_f_all[:adam_epochs], 'g-', alpha=0.8)
    if history_lbfgs['lam_bc']:
        ax.plot(range(adam_epochs, total_iterations), lam_bc_all[adam_epochs:], 
                'b--', alpha=0.8)
        ax.plot(range(adam_epochs, total_iterations), lam_data_all[adam_epochs:], 
                'orange', linestyle='--', alpha=0.8)
        ax.plot(range(adam_epochs, total_iterations), lam_f_all[adam_epochs:], 
                'g--', alpha=0.8)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', label='λ_bc')
    ax.plot([], [], 'orange', label='λ_data')
    ax.plot([], [], 'g-', label='λ_f')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weight')
    ax.set_title('IDW Weights (solid=Adam, dashed=L-BFGS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Lambda weights (log)
    ax = axes[1, 2]
    ax.semilogy(range(adam_epochs), lam_bc_all[:adam_epochs], 'b-', alpha=0.8)
    ax.semilogy(range(adam_epochs), lam_data_all[:adam_epochs], 'orange', alpha=0.8)
    ax.semilogy(range(adam_epochs), lam_f_all[:adam_epochs], 'g-', alpha=0.8)
    if history_lbfgs['lam_bc']:
        ax.semilogy(range(adam_epochs, total_iterations), lam_bc_all[adam_epochs:], 
                    'b--', alpha=0.8)
        ax.semilogy(range(adam_epochs, total_iterations), lam_data_all[adam_epochs:], 
                    'orange', linestyle='--', alpha=0.8)
        ax.semilogy(range(adam_epochs, total_iterations), lam_f_all[adam_epochs:], 
                    'g--', alpha=0.8)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', label='λ_bc')
    ax.plot([], [], 'orange', label='λ_data')
    ax.plot([], [], 'g-', label='λ_f')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weight (log)')
    ax.set_title('IDW Weights Log (solid=Adam, dashed=L-BFGS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    print(f"Saved: {filepath}")
    plt.close()
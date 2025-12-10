"""
Main training script for 2D diffusion inverse problem with IDW-PINN.

Orchestrates:
- Configuration loading
- Data preparation
- Model initialization
- Two-phase training (Adam + L-BFGS)
- Visualization and results summary

Extracted from legacy: 2D_num_inv_IDW_newPrintOut_newFigs.py (main code section)
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from idw_pinn.config import Config
from idw_pinn.data import load_2d_diffusion_data
from idw_pinn.models import PINN
from idw_pinn.training import IDWPINNTrainer
from idw_pinn.utils import plot_2d_solution_comparison, plot_training_diagnostics


def setup_environment(seed=123):
    """
    Configure reproducibility and TensorFlow environment.
    
    Extracted from legacy: lines immediately after imports
    """
    # Disable oneDNN custom operations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    print(f"TensorFlow version: {tf.__version__}")


def print_header(config):
    """Print formatted header with problem description."""
    print("\n" + "="*70)
    print("2D Diffusion Inverse Problem with IDW Weighting")
    print("="*70)
    print(f"\nConfiguration: {config.data.input_file}")
    print(f"True D = {config.physics.diff_coeff_true}")
    print(f"Initial D guess = {config.physics.diff_coeff_init}")


def print_final_summary(config, data, model, training_results, output_dir='outputs'):
    """
    Print comprehensive final summary matching legacy format.
    
    Extracted from legacy: lines ~820-850 (final summary block)
    """
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # Configuration
    print("\n--- Configuration ---")
    print(f"  DIFF_COEFF_TRUE     = {config.physics.diff_coeff_true}")
    print(f"  DIFF_COEFF_INIT     = {config.physics.diff_coeff_init}")
    print(f"  IDW_EMA_BETA        = {config.idw.ema_beta}")
    print(f"  IDW_EPS             = {config.idw.eps}")
    print(f"  IDW_CLAMP           = ({config.idw.clamp_min}, {config.idw.clamp_max})")
    print(f"  WEIGHT_SUM_TARGET   = {config.idw.weight_sum_target}")
    print(f"  FREEZE_BEFORE_LBFGS = {config.idw.freeze_before_lbfgs}")
    print(f"  ADAM_LR             = {config.training.adam_lr}")
    print(f"  ADAM_EPOCHS         = {config.training.adam_epochs}")
    
    # Data
    print("\n--- Data ---")
    x, y, t = data['grid']
    print(f"  Input file          = {config.data.input_file}")
    print(f"  nx={len(x)}, ny={len(y)}, nt={len(t)}")
    print(f"  Domain: x in [{x.min():.2f},{x.max():.2f}], "
          f"y in [{y.min():.2f},{y.max():.2f}], "
          f"t in [{t.min():.2f},{t.max():.2f}]")
    print(f"  N_u (BC/IC points)  = {config.data.n_u}")
    print(f"  N_f (collocation)   = {config.data.n_f}")
    print(f"  N_obs (interior)    = {config.data.n_obs}")
    
    # Network
    print("\n--- Network ---")
    print(f"  Layers              = {config.network.layers}")
    print(f"  Total parameters    = {model.parameters + 1}")  # +1 for diff_coeff
    
    # Training Time
    print("\n--- Training Time ---")
    print(f"  Adam time           = {training_results['adam_time']:.2f}s")
    print(f"  L-BFGS time         = {training_results['lbfgs_time']:.2f}s")
    print(f"  Total time          = {training_results['adam_time'] + training_results['lbfgs_time']:.2f}s")
    print(f"  L-BFGS iterations   = {training_results['lbfgs_results'].nit}")
    print(f"  L-BFGS func evals   = {training_results['lbfgs_results'].nfev}")
    
    lbfgs_msg = training_results['lbfgs_results'].message
    if isinstance(lbfgs_msg, bytes):
        lbfgs_msg = lbfgs_msg.decode()
    print(f"  L-BFGS termination  = {lbfgs_msg}")
    
    # Results
    print("\n--- Results ---")
    print(f"  D_true              = {data['diff_coeff_true']}")
    print(f"  D_learned           = {training_results['diff_coeff_learned']:.6f}")
    print(f"  D_error             = {training_results['diff_coeff_error']:.6f} "
          f"({100*training_results['diff_coeff_error']/data['diff_coeff_true']:.2f}%)")
    print(f"  Relative L2 error   = {training_results['final_error']:.5e}")
    
    # Final IDW Weights
    print("\n--- Final IDW Weights ---")
    final_lam_bc = training_results['history_lbfgs']['lam_bc'][-1] if training_results['history_lbfgs']['lam_bc'] else training_results['history']['lam_bc'][-1]
    final_lam_data = training_results['history_lbfgs']['lam_data'][-1] if training_results['history_lbfgs']['lam_data'] else training_results['history']['lam_data'][-1]
    final_lam_f = training_results['history_lbfgs']['lam_f'][-1] if training_results['history_lbfgs']['lam_f'] else training_results['history']['lam_f'][-1]
    print(f"  lambda_bc           = {final_lam_bc:.6f}")
    print(f"  lambda_data         = {final_lam_data:.6f}")
    print(f"  lambda_f            = {final_lam_f:.6f}")
    
    # Output Files
    print("\n--- Output Files ---")
    print(f"  {output_dir}/diff2D_IDW_inverse.png")
    print(f"  {output_dir}/inverse_diagnostics_2D.png")
    
    print("="*70)


def main():
    """Main execution flow."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train 2D diffusion inverse PINN')
    parser.add_argument('--config', type=str, 
                    default='../configs/default_2d_inverse.yaml',  # ← relative to scripts/
                    help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory for output files')
    args = parser.parse_args()
    
    # Setup
    setup_environment(seed=123)
    
    # Load configuration
    config = Config(args.config)
    print_header(config)
    
    # Load data
    print("\n--- Loading Data ---")
    data = load_2d_diffusion_data(config)
    print(f"Loaded {config.data.input_file}")
    print(f"  Ground truth D = {data['diff_coeff_true']}")
    print(f"  Training points: BC/IC={data['X_u_train'].shape[0]}, "
          f"Observations={data['X_obs'].shape[0]}, "
          f"Collocation={data['X_f_train'].shape[0]}")
    
    # Create model
    print("\n--- Initializing Model ---")
    lb, ub = data['bounds']
    model = PINN(
        layers=config.network.layers,
        lb=lb,
        ub=ub,
        diff_coeff_init=config.physics.diff_coeff_init,
        idw_config=config
    )
    print(f"Network: {config.network.layers}")
    print(f"Total parameters: {model.parameters + 1}")  # +1 for diff_coeff
    
    # Create trainer
    print("\n--- Initializing Trainer ---")
    trainer = IDWPINNTrainer(model, config, data)
    
    # Train
    print("\n--- Starting Training ---")
    training_results = trainer.train()
    
    # Generate visualizations
    print("\n--- Generating Visualizations ---")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Solution comparison plot
    u_pred = model.evaluate(data['X_u_test']).numpy()
    x, y, t = data['grid']
    plot_2d_solution_comparison(
        u_pred=u_pred,
        usol=data['usol'],
        x=x,
        y=y,
        t=t,
        diff_coeff_learned=training_results['diff_coeff_learned'],
        diff_coeff_true=data['diff_coeff_true'],
        output_dir=args.output_dir,
        filename='diff2D_IDW_inverse.png'
    )
    
    # Training diagnostics plot
    plot_training_diagnostics(
        history_adam=training_results['history'],
        history_lbfgs=training_results['history_lbfgs'],
        diff_coeff_true=data['diff_coeff_true'],
        output_dir=args.output_dir,
        filename='inverse_diagnostics_2D.png'
    )
    
    # Print comprehensive summary
    print_final_summary(config, data, model, training_results, args.output_dir)


if __name__ == "__main__":
    main()
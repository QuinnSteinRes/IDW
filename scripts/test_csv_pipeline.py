#!/usr/bin/env python
"""
Test script for CSV data pipeline verification.

This script:
1. Generates synthetic CSV data with known diffusion coefficient
2. Loads it using the CSV loader
3. Verifies the data structure matches expectations
4. Optionally runs a quick training test

Usage:
    python scripts/test_csv_pipeline.py [--train]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))


def test_synthetic_generation():
    """Test synthetic CSV data generation."""
    print("\n" + "="*60)
    print("TEST 1: Synthetic CSV Data Generation")
    print("="*60)
    
    from generate_synthetic_csv import generate_synthetic_csv, analyze_csv_structure
    
    # Generate with known parameters
    output_path = 'data/test_synthetic.csv'
    D_true = 0.001
    
    df, metadata = generate_synthetic_csv(
        nx=50,  # Smaller grid for quick test
        ny=30,
        nt=5,
        x_start=100,
        y_start=200,
        diff_coeff=D_true,
        ic_type='gaussian',
        output_path=output_path,
        save_mat=True
    )
    
    # Verify structure
    print("\n--- Verification ---")
    assert len(df) == 50 * 30 * 5, f"Expected {50*30*5} points, got {len(df)}"
    assert list(df.columns) == ['x', 'y', 't', 'intensity'], f"Wrong columns: {df.columns}"
    assert df['x'].nunique() == 50, "Wrong number of unique x values"
    assert df['y'].nunique() == 30, "Wrong number of unique y values"
    assert df['t'].nunique() == 5, "Wrong number of unique t values"
    assert metadata['diff_coeff_true'] == D_true, "Metadata D mismatch"
    
    print("✓ Synthetic generation passed!")
    return output_path, D_true


def test_csv_loader(csv_path, D_true):
    """Test CSV data loading."""
    print("\n" + "="*60)
    print("TEST 2: CSV Data Loading")
    print("="*60)
    
    # Create minimal config object
    class MinimalConfig:
        class data:
            input_file = csv_path
            n_u = 100
            n_f = 500
            n_obs = 200
            normalize_coords = True
            normalize_intensity = True
        
        class physics:
            diff_coeff_true = D_true
    
    config = MinimalConfig()
    
    from idw_pinn.data.csv_loader import load_csv_diffusion_data
    
    data = load_csv_diffusion_data(config)
    
    # Verify loaded data structure
    print("\n--- Verification ---")
    
    # Check all required keys present
    required_keys = ['X_f_train', 'X_u_train', 'u_train', 'X_obs', 'u_obs',
                     'X_u_test', 'u_test', 'bounds', 'grid', 'usol', 
                     'diff_coeff_true', 'metadata']
    for key in required_keys:
        assert key in data, f"Missing key: {key}"
    
    # Check shapes
    assert data['X_u_train'].shape[1] == 3, "X_u_train should have 3 columns (x,y,t)"
    assert data['u_train'].shape[1] == 1, "u_train should have 1 column"
    assert data['X_obs'].shape[1] == 3, "X_obs should have 3 columns"
    assert len(data['bounds']) == 2, "bounds should be (lb, ub) tuple"
    assert data['usol'].ndim == 3, "usol should be 3D (nx, ny, nt)"
    
    # Check normalization
    lb, ub = data['bounds']
    assert np.allclose(lb, [0, 0, 0]), f"Lower bound should be [0,0,0], got {lb}"
    assert np.allclose(ub, [1, 1, 1]), f"Upper bound should be [1,1,1], got {ub}"
    
    # Check ground truth loaded
    assert data['diff_coeff_true'] == D_true, f"D mismatch: {data['diff_coeff_true']} vs {D_true}"
    
    print(f"✓ X_f_train shape: {data['X_f_train'].shape}")
    print(f"✓ X_u_train shape: {data['X_u_train'].shape}")
    print(f"✓ X_obs shape: {data['X_obs'].shape}")
    print(f"✓ usol shape: {data['usol'].shape}")
    print(f"✓ Ground truth D: {data['diff_coeff_true']}")
    print("✓ CSV loading passed!")
    
    return data


def test_quick_training(data, D_true):
    """Quick training test to verify PINN can be instantiated and trained."""
    print("\n" + "="*60)
    print("TEST 3: Quick Training Verification")
    print("="*60)
    
    try:
        import tensorflow as tf
        from idw_pinn.models import PINN
        from idw_pinn.training import IDWPINNTrainer
        
        # Suppress TF warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.random.set_seed(123)
        np.random.seed(123)
        
        # Create minimal config for quick training
        class QuickConfig:
            class physics:
                diff_coeff_true = D_true
                diff_coeff_init = D_true * 5  # Start far from true
            
            class idw:
                ema_beta = 0.9
                eps = 1e-12
                clamp_min = 1e-3
                clamp_max = 1e3
                weight_sum_target = 3.0
                freeze_before_lbfgs = True
            
            class training:
                adam_lr = 1e-3
                adam_epochs = 100  # Very few for test
                print_every = 50
                lbfgs_maxiter = 50
                lbfgs_maxfun = 100
                lbfgs_maxcor = 50
                lbfgs_ftol = 1e-9
                lbfgs_gtol = 1e-5
                lbfgs_maxls = 20
            
            class network:
                layers = [3, 32, 32, 1]  # Small network for quick test
        
        config = QuickConfig()
        
        # Create model
        lb, ub = data['bounds']
        model = PINN(
            layers=config.network.layers,
            lb=lb,
            ub=ub,
            diff_coeff_init=config.physics.diff_coeff_init,
            idw_config=config
        )
        
        print(f"Model created with {model.parameters + 1} parameters")
        print(f"Initial D guess: {config.physics.diff_coeff_init}")
        
        # Create trainer
        trainer = IDWPINNTrainer(model, config, data)
        
        # Run quick training
        print("\nRunning quick training (100 Adam epochs + 50 L-BFGS)...")
        results = trainer.train()
        
        D_learned = results['diff_coeff_learned']
        D_error = abs(D_learned - D_true)
        
        print(f"\n--- Results ---")
        print(f"D_true:    {D_true:.6f}")
        print(f"D_learned: {D_learned:.6f}")
        print(f"D_error:   {D_error:.6f} ({100*D_error/D_true:.1f}%)")
        
        # For a quick test, we just verify training ran without errors
        # Full convergence would require many more epochs
        print("✓ Quick training completed without errors!")
        
        return results
        
    except ImportError as e:
        print(f"⚠ Skipping training test - missing dependency: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Test CSV data pipeline')
    parser.add_argument('--train', action='store_true',
                        help='Include quick training test')
    parser.add_argument('--skip-generation', action='store_true',
                        help='Skip synthetic data generation (use existing)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CSV PIPELINE VERIFICATION")
    print("="*60)
    
    # Test 1: Generate synthetic data
    if args.skip_generation:
        csv_path = 'data/test_synthetic.csv'
        D_true = 0.001
        print(f"\nUsing existing data: {csv_path}")
    else:
        csv_path, D_true = test_synthetic_generation()
    
    # Test 2: Load CSV data
    data = test_csv_loader(csv_path, D_true)
    
    # Test 3: Quick training (optional)
    if args.train:
        test_quick_training(data, D_true)
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print(f"\nCSV pipeline is ready for use.")
    print(f"Generated files:")
    print(f"  - {csv_path}")
    print(f"  - {csv_path.replace('.csv', '.mat')}")
    print(f"  - {csv_path.replace('.csv', '.json')}")


if __name__ == "__main__":
    main()

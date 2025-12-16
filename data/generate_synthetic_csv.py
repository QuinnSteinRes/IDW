"""
Generate synthetic diffusion data in CSV format matching experimental structure.

This script creates a synthetic dataset that mimics the structure of experimental
microscopy data (x, y, t, intensity columns) while solving the diffusion equation
with a known diffusion coefficient. This allows verification of the CSV data
pipeline before using real experimental data.

Experimental data structure (from intensity_time_series_spatial_temporal.csv):
- x: pixel coordinates (e.g., 2450-2749, 300 unique values)
- y: pixel coordinates (e.g., 800-899, 100 unique values)
- t: frame numbers (e.g., 0-7, 8 unique values)
- intensity: grayscale values (e.g., 10-204)

The synthetic data uses the same grid structure but with a known initial condition
and diffusion coefficient, enabling ground-truth validation of inverse problem results.

Usage:
    python data/generate_synthetic_csv.py [--output OUTPUT] [--diff-coeff D]
    
Example:
    python data/generate_synthetic_csv.py --output data/synthetic_diffusion.csv --diff-coeff 0.001
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def solve_diffusion_2d(nx, ny, nt, dx, dy, dt, D, u0):
    """
    Solve 2D diffusion equation using Forward Euler (explicit) method.
    
    PDE: du/dt = D * (d²u/dx² + d²u/dy²)
    
    Parameters:
    -----------
    nx, ny : int
        Number of grid points in x and y
    nt : int
        Number of time steps
    dx, dy : float
        Grid spacing in x and y
    dt : float
        Time step size
    D : float
        Diffusion coefficient
    u0 : ndarray
        Initial condition, shape (nx, ny)
        
    Returns:
    --------
    u : ndarray
        Solution array of shape (nt, nx, ny)
    """
    # Check stability: Fourier number criterion for 2D
    Fx = D * dt / dx**2
    Fy = D * dt / dy**2
    if Fx + Fy >= 0.5:
        print(f"Warning: Stability criterion violated! Fx + Fy = {Fx + Fy:.4f} >= 0.5")
        print(f"  Consider reducing dt or D, or increasing dx/dy")
    
    # Initialize solution array
    u = np.zeros((nt, nx, ny))
    u[0, :, :] = u0
    
    # Time stepping
    for n in range(nt - 1):
        # Interior points (vectorized)
        u[n+1, 1:-1, 1:-1] = (u[n, 1:-1, 1:-1] + 
                              Fx * (u[n, 0:-2, 1:-1] - 2*u[n, 1:-1, 1:-1] + u[n, 2:, 1:-1]) +
                              Fy * (u[n, 1:-1, 0:-2] - 2*u[n, 1:-1, 1:-1] + u[n, 1:-1, 2:]))
        
        # Neumann boundary conditions (zero flux) - common for experimental ROI
        # This is more realistic than Dirichlet for an ROI extracted from a larger image
        u[n+1, 0, :] = u[n+1, 1, :]      # left boundary
        u[n+1, -1, :] = u[n+1, -2, :]    # right boundary
        u[n+1, :, 0] = u[n+1, :, 1]      # bottom boundary
        u[n+1, :, -1] = u[n+1, :, -2]    # top boundary
    
    return u


def create_initial_condition(nx, ny, x, y, ic_type='gaussian'):
    """
    Create initial condition for diffusion simulation.
    
    Parameters:
    -----------
    nx, ny : int
        Grid dimensions
    x, y : ndarray
        Coordinate arrays
    ic_type : str
        Type of initial condition: 'gaussian', 'sine', 'step', 'experimental_like'
        
    Returns:
    --------
    u0 : ndarray
        Initial condition array, shape (nx, ny)
    """
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Normalize coordinates to [0, 1] for IC definition
    x_norm = (X - X.min()) / (X.max() - X.min())
    y_norm = (Y - Y.min()) / (Y.max() - Y.min())
    
    if ic_type == 'gaussian':
        # Single Gaussian blob in center
        cx, cy = 0.5, 0.5
        sigma = 0.15
        u0 = np.exp(-((x_norm - cx)**2 + (y_norm - cy)**2) / (2 * sigma**2))
        
    elif ic_type == 'sine':
        # Product of sines (smooth, known analytical behavior)
        u0 = np.sin(np.pi * x_norm) * np.sin(np.pi * y_norm)
        
    elif ic_type == 'step':
        # Step function (sharp gradient, tests high-frequency capture)
        u0 = np.where((x_norm > 0.3) & (x_norm < 0.7) & 
                      (y_norm > 0.3) & (y_norm < 0.7), 1.0, 0.0)
        
    elif ic_type == 'experimental_like':
        # Mimics dye diffusion: bright spot that spreads
        # Multiple Gaussian blobs to simulate realistic dye distribution
        u0 = np.zeros_like(X, dtype=np.float64)
        
        # Main concentration region
        cx, cy = 0.4, 0.5
        sigma = 0.12
        u0 += 0.8 * np.exp(-((x_norm - cx)**2 + (y_norm - cy)**2) / (2 * sigma**2))
        
        # Secondary region
        cx2, cy2 = 0.65, 0.45
        sigma2 = 0.08
        u0 += 0.5 * np.exp(-((x_norm - cx2)**2 + (y_norm - cy2)**2) / (2 * sigma2**2))
        
        # Add slight background gradient (common in experimental data)
        u0 += 0.1 * x_norm
        
    else:
        raise ValueError(f"Unknown IC type: {ic_type}")
    
    return u0


def solution_to_csv(u, x_coords, y_coords, t_coords, 
                    intensity_min=10, intensity_max=204,
                    x_offset=0, y_offset=0):
    """
    Convert solution array to CSV format matching experimental structure.
    
    Parameters:
    -----------
    u : ndarray
        Solution array, shape (nt, nx, ny)
    x_coords : ndarray
        X coordinate values (can be pixel indices)
    y_coords : ndarray
        Y coordinate values (can be pixel indices)
    t_coords : ndarray
        Time/frame values
    intensity_min, intensity_max : int
        Range for intensity scaling (to match experimental data range)
    x_offset, y_offset : int
        Offset to add to coordinates (to match experimental pixel positions)
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with columns [x, y, t, intensity]
    """
    nt, nx, ny = u.shape
    
    # Normalize solution to [0, 1] then scale to intensity range
    u_min, u_max = u.min(), u.max()
    if u_max > u_min:
        u_normalized = (u - u_min) / (u_max - u_min)
    else:
        u_normalized = np.zeros_like(u)
    
    # Scale to integer intensity range
    intensity_scaled = intensity_min + (intensity_max - intensity_min) * u_normalized
    intensity_int = np.round(intensity_scaled).astype(int)
    
    # Build DataFrame row by row (matching experimental CSV structure)
    data = []
    for ti, t_val in enumerate(t_coords):
        for xi, x_val in enumerate(x_coords):
            for yi, y_val in enumerate(y_coords):
                data.append({
                    'x': int(x_val + x_offset),
                    'y': int(y_val + y_offset),
                    't': int(t_val),
                    'intensity': intensity_int[ti, xi, yi]
                })
    
    df = pd.DataFrame(data)
    return df


def generate_synthetic_csv(
    # Grid parameters (matching experimental structure)
    nx=300,
    ny=100,
    nt=8,
    # Coordinate ranges (matching experimental pixel coordinates)
    x_start=2450,
    y_start=800,
    # Physical parameters
    diff_coeff=0.001,  # Diffusion coefficient in pixel²/frame units
    # Initial condition
    ic_type='experimental_like',
    # Output scaling
    intensity_min=10,
    intensity_max=204,
    # Output file
    output_path='synthetic_diffusion.csv',
    # Also save ground truth .mat file for comparison
    save_mat=True
):
    """
    Generate synthetic diffusion CSV data matching experimental structure.
    
    Parameters:
    -----------
    nx, ny, nt : int
        Grid dimensions (spatial and temporal)
    x_start, y_start : int
        Starting pixel coordinates (to match experimental ROI position)
    diff_coeff : float
        True diffusion coefficient (in pixel²/frame units)
    ic_type : str
        Initial condition type
    intensity_min, intensity_max : int
        Intensity range for output
    output_path : str
        Path for output CSV file
    save_mat : bool
        Whether to also save .mat file with ground truth
        
    Returns:
    --------
    df : pandas.DataFrame
        Generated data
    metadata : dict
        Simulation metadata including true diffusion coefficient
    """
    print("="*60)
    print("Generating Synthetic Diffusion Data (CSV Format)")
    print("="*60)
    
    # Create coordinate arrays
    x = np.arange(nx)  # 0 to nx-1, will add offset later
    y = np.arange(ny)  # 0 to ny-1, will add offset later
    t = np.arange(nt)  # Frame numbers
    
    # Grid spacing (1 pixel, 1 frame)
    dx = 1.0
    dy = 1.0
    dt = 1.0
    
    # Check stability and compute appropriate dt for simulation
    # For explicit scheme: D * dt / dx² < 0.25 (conservative for 2D)
    max_dt = 0.25 * min(dx**2, dy**2) / diff_coeff
    
    if dt > max_dt:
        # Need to use substeps for stability
        n_substeps = int(np.ceil(dt / max_dt))
        dt_sim = dt / n_substeps
        print(f"Using {n_substeps} substeps per frame for stability")
    else:
        n_substeps = 1
        dt_sim = dt
    
    print(f"\nGrid: {nx} x {ny} x {nt} (x, y, t)")
    print(f"Coordinates: x=[{x_start}, {x_start+nx-1}], y=[{y_start}, {y_start+ny-1}], t=[0, {nt-1}]")
    print(f"Diffusion coefficient: D = {diff_coeff} pixel²/frame")
    print(f"Initial condition: {ic_type}")
    
    # Create initial condition
    u0 = create_initial_condition(nx, ny, x, y, ic_type)
    print(f"Initial intensity range: [{u0.min():.3f}, {u0.max():.3f}]")
    
    # Solve diffusion equation
    # If we need substeps, solve at higher temporal resolution then subsample
    if n_substeps > 1:
        nt_sim = (nt - 1) * n_substeps + 1
        u_fine = solve_diffusion_2d(nx, ny, nt_sim, dx, dy, dt_sim, diff_coeff, u0)
        # Subsample to original frame rate
        u = u_fine[::n_substeps, :, :]
    else:
        u = solve_diffusion_2d(nx, ny, nt, dx, dy, dt_sim, diff_coeff, u0)
    
    print(f"Solution computed: shape = {u.shape}")
    print(f"Final intensity range: [{u[-1].min():.3f}, {u[-1].max():.3f}]")
    
    # Convert to CSV format
    df = solution_to_csv(
        u, x, y, t,
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        x_offset=x_start,
        y_offset=y_start
    )
    
    print(f"\nCSV data shape: {df.shape}")
    print(f"Expected shape: ({nx * ny * nt}, 4) = ({nx * ny * nt}, 4)")
    
    # Save CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved CSV: {output_path}")
    
    # Metadata for ground truth comparison
    metadata = {
        'diff_coeff_true': diff_coeff,
        'nx': nx,
        'ny': ny,
        'nt': nt,
        'x_start': x_start,
        'y_start': y_start,
        'dx': dx,
        'dy': dy,
        'dt': dt,
        'ic_type': ic_type,
        'intensity_min': intensity_min,
        'intensity_max': intensity_max,
        'boundary_conditions': 'neumann'  # Zero-flux, realistic for ROI
    }
    
    # Optionally save .mat file for ground truth comparison
    if save_mat:
        import scipy.io
        mat_path = output_path.with_suffix('.mat')
        mat_data = {
            'usol': u,  # Shape: (nt, nx, ny)
            'x': x + x_start,
            'y': y + y_start,
            't': t,
            'diffCoeff': diff_coeff,
            'ic_type': ic_type
        }
        scipy.io.savemat(mat_path, mat_data)
        print(f"Saved MAT: {mat_path}")
    
    # Save metadata as JSON
    import json
    meta_path = output_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_path}")
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"\nGround truth diffusion coefficient: D = {diff_coeff}")
    print(f"Use this value to validate inverse problem results.")
    
    return df, metadata


def analyze_csv_structure(csv_path):
    """
    Analyze CSV file structure for comparison with experimental data.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file
    """
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*60)
    print(f"CSV Analysis: {csv_path}")
    print("="*60)
    
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    print(f"\nColumn ranges:")
    for col in df.columns:
        print(f"  {col}: [{df[col].min()}, {df[col].max()}], "
              f"unique={df[col].nunique()}")
    
    # Check grid structure
    nx = df['x'].nunique()
    ny = df['y'].nunique()
    nt = df['t'].nunique()
    expected = nx * ny * nt
    
    print(f"\nGrid structure: {nx} x {ny} x {nt}")
    print(f"Expected points: {expected}")
    print(f"Actual points: {len(df)}")
    print(f"Complete: {len(df) == expected}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate synthetic diffusion data in CSV format'
    )
    parser.add_argument('--output', type=str, 
                        default='data/synthetic_diffusion.csv',
                        help='Output CSV file path')
    parser.add_argument('--diff-coeff', type=float, default=0.001,
                        help='True diffusion coefficient (pixel²/frame)')
    parser.add_argument('--nx', type=int, default=300,
                        help='Number of x grid points')
    parser.add_argument('--ny', type=int, default=100,
                        help='Number of y grid points')
    parser.add_argument('--nt', type=int, default=8,
                        help='Number of time frames')
    parser.add_argument('--ic-type', type=str, default='experimental_like',
                        choices=['gaussian', 'sine', 'step', 'experimental_like'],
                        help='Initial condition type')
    parser.add_argument('--analyze', type=str, default=None,
                        help='Analyze existing CSV file instead of generating')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_csv_structure(args.analyze)
    else:
        generate_synthetic_csv(
            nx=args.nx,
            ny=args.ny,
            nt=args.nt,
            diff_coeff=args.diff_coeff,
            ic_type=args.ic_type,
            output_path=args.output
        )
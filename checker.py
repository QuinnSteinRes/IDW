"""
Script to analyze experimental CSV data structure.
Run this in your environment where the CSV file is accessible.

Usage:
    python analyze_csv.py path/to/intensity_time_series_spatial_temporal.csv
"""
import sys
import pandas as pd
import numpy as np

def analyze_csv(filepath):
    """Analyze CSV structure and print key information."""
    
    print("="*70)
    print("EXPERIMENTAL CSV DATA ANALYSIS")
    print("="*70)
    
    # Load the data
    df = pd.read_csv(filepath)
    
    # Basic info
    print("\n--- Basic Structure ---")
    print(f"Shape: {df.shape} (rows, columns)")
    print(f"Column names: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    # First few rows
    print("\n--- First 10 rows ---")
    print(df.head(10).to_string())
    
    # Last few rows
    print("\n--- Last 5 rows ---")
    print(df.tail(5).to_string())
    
    # Statistics for each column
    print("\n--- Column Statistics ---")
    for col in df.columns:
        print(f"\n{col}:")
        print(f"  min: {df[col].min()}")
        print(f"  max: {df[col].max()}")
        print(f"  mean: {df[col].mean():.6f}")
        print(f"  std: {df[col].std():.6f}")
        print(f"  unique values: {df[col].nunique()}")
        
    # Grid structure analysis
    print("\n--- Grid Structure Analysis ---")
    
    # Check for x, y, t columns (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    
    x_col = col_map.get('x')
    y_col = col_map.get('y')
    t_col = col_map.get('t') or col_map.get('time')
    intensity_col = col_map.get('intensity') or col_map.get('i')
    
    if x_col:
        x_unique = np.sort(df[x_col].unique())
        print(f"\nX coordinates:")
        print(f"  Number of unique values: {len(x_unique)}")
        print(f"  Range: [{x_unique.min()}, {x_unique.max()}]")
        if len(x_unique) > 1:
            dx = np.diff(x_unique)
            print(f"  Spacing (dx): min={dx.min():.6f}, max={dx.max():.6f}, mean={dx.mean():.6f}")
            print(f"  First 5 unique x: {x_unique[:5]}")
            print(f"  Last 5 unique x: {x_unique[-5:]}")
    
    if y_col:
        y_unique = np.sort(df[y_col].unique())
        print(f"\nY coordinates:")
        print(f"  Number of unique values: {len(y_unique)}")
        print(f"  Range: [{y_unique.min()}, {y_unique.max()}]")
        if len(y_unique) > 1:
            dy = np.diff(y_unique)
            print(f"  Spacing (dy): min={dy.min():.6f}, max={dy.max():.6f}, mean={dy.mean():.6f}")
            print(f"  First 5 unique y: {y_unique[:5]}")
            print(f"  Last 5 unique y: {y_unique[-5:]}")
    
    if t_col:
        t_unique = np.sort(df[t_col].unique())
        print(f"\nTime coordinates:")
        print(f"  Number of unique values (frames): {len(t_unique)}")
        print(f"  Range: [{t_unique.min()}, {t_unique.max()}]")
        if len(t_unique) > 1:
            dt = np.diff(t_unique)
            print(f"  Spacing (dt): min={dt.min():.6f}, max={dt.max():.6f}, mean={dt.mean():.6f}")
            print(f"  First 5 unique t: {t_unique[:5]}")
            print(f"  Last 5 unique t: {t_unique[-5:]}")
    
    if intensity_col:
        print(f"\nIntensity values:")
        print(f"  Range: [{df[intensity_col].min()}, {df[intensity_col].max()}]")
        print(f"  Mean: {df[intensity_col].mean():.6f}")
        print(f"  Std: {df[intensity_col].std():.6f}")
        
    # Check data completeness
    print("\n--- Data Completeness ---")
    if x_col and y_col and t_col:
        nx = df[x_col].nunique()
        ny = df[y_col].nunique()
        nt = df[t_col].nunique()
        expected_points = nx * ny * nt
        actual_points = len(df)
        print(f"  Expected points (nx * ny * nt): {nx} * {ny} * {nt} = {expected_points}")
        print(f"  Actual points: {actual_points}")
        print(f"  Data is {'COMPLETE' if expected_points == actual_points else 'SPARSE/INCOMPLETE'}")
        if expected_points != actual_points:
            print(f"  Missing points: {expected_points - actual_points}")
            print(f"  Coverage: {100*actual_points/expected_points:.2f}%")
    
    # Check for NaN values
    print("\n--- Missing Values ---")
    nan_counts = df.isna().sum()
    print(nan_counts.to_string())
    
    # Units hint (based on value ranges)
    print("\n--- Units Hint (based on value ranges) ---")
    if x_col and y_col:
        x_range = df[x_col].max() - df[x_col].min()
        y_range = df[y_col].max() - df[y_col].min()
        print(f"  X range: {x_range:.4f}")
        print(f"  Y range: {y_range:.4f}")
        if x_range < 10 and y_range < 10:
            print("  --> Likely in physical units (mm or similar)")
        elif x_range > 100:
            print("  --> Likely in pixel units")
    
    if t_col:
        t_range = df[t_col].max() - df[t_col].min()
        print(f"  T range: {t_range:.4f}")
        if t_range < 100:
            print("  --> Could be in seconds or frame numbers")
        else:
            print("  --> Likely frame numbers")
    
    print("\n" + "="*70)
    print("END OF ANALYSIS")
    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_csv.py <path_to_csv>")
        print("\nExample:")
        print("  python analyze_csv.py src/idw_pinn/data/intensity_time_series_spatial_temporal.csv")
        sys.exit(1)
    
    filepath = sys.argv[1]
    analyze_csv(filepath)

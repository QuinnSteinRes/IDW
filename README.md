# IDW-PINN: Inverse-Dirichlet Weighted Physics-Informed Neural Networks

Implementation of Physics-Informed Neural Networks with Inverse-Dirichlet Weighting for solving inverse problems in diffusion equations.

## Installation

```bash
pip install -e .
```

## Usage

```bash
python scripts/train_2d_inverse.py --config configs/default_2d_inverse.yaml
```

## Repository Structure

```
\u251c\u2500\u2500 configs/              # YAML configuration files
\u251c\u2500\u2500 data/                 # Ground truth data (.mat files)
\u251c\u2500\u2500 outputs/              # Generated figures and logs
\u251c\u2500\u2500 scripts/              # Entry point scripts
\u2514\u2500\u2500 src/idw_pinn/
    \u251c\u2500\u2500 config.py         # Configuration loader
    \u251c\u2500\u2500 models/           # PINN model definition
    \u251c\u2500\u2500 data/             # Data loading utilities
    \u251c\u2500\u2500 training/         # Training loops and IDW weighting
    \u251c\u2500\u2500 losses/           # PDE loss functions
    \u2514\u2500\u2500 utils/            # Visualization and helpers
```

## References

- Maddu et al. (2021) "Inverse-Dirichlet Weighting Enables Reliable Training of Physics Informed Neural Networks"

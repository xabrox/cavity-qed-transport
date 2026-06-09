# Two-Quantum-Dot Rabi Transport Model (Strong Coupling) — Python

Pure-Python version of the double-quantum-dot / cavity transport model in the
**strong / deep-strong light–matter coupling** regime, solved with a Lindblad
master / rate equation in the polaron (displaced-Fock) basis.

The model uses a quantum-Rabi dot–cavity coupling, dresses lead-tunneling and
inter-dot hopping with displacement operators (Franck–Condon physics), and computes
steady-state currents and `I–V` characteristics. A Jaynes–Cummings (rotating-wave)
builder is included for spectrum and transport comparison against the full Rabi model.

## Layout

| File | Description |
|------|-------------|
| `dot_transport.py` | **Model library** — parameter containers, operators, `HamiltonianBuilder`, master-equation matrices, rate solver, `TransportCalculator`. Importable, no side effects. |
| `analysis.py` | **Studies & figures** (§6–§12) — imports the library, reproduces all analyses, writes PNGs to `figures/`. |
| `jc_wc.py` | Jaynes–Cummings / weak-coupling helper used by the comparison studies. |
| `requirements.txt` | Python dependencies (`numpy`, `scipy`, `matplotlib`). |

## Model library (`dot_transport.py`)

- `DotParameters`, `CavityParameters`, `LeadParameters` — model parameters
- `PhotonicOperators`, `ElectronOperators`, `TunnelOperators`, `DisplacementOperator` — operators in the Fock ⊗ dot basis
- `HamiltonianBuilder` — full quantum-Rabi Hamiltonian (`.diagonalize()`)
- `CavityDecayMatrix`, `TunnelRateMatrix`, `RateEquationSolver` — Lindblad/rate master equation
- `TransportCalculator` — steady-state current

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Reproduce all figures (written to `figures/`):

```bash
python analysis.py
```

Use the model as a library:

```python
from dot_transport import DotParameters, CavityParameters, HamiltonianBuilder

dots   = DotParameters(e1=0.6, e2=0.6, t=0.5)
cavity = CavityParameters(n=6, omega=1.0, kappa=0.0)
eigvals, eigvecs, _ = HamiltonianBuilder(cavity, dots, lam=0.3).diagonalize()
```

## Notes

- **Coupling convention:** the effective Jaynes–Cummings coupling is `g_JC = t · λ`,
  with `t` the inter-dot hopping and `λ` the dimensionless light–matter coupling.
- `analysis.py` both saves figures to `figures/` and calls `plt.show()`; run with a
  headless backend (`MPLBACKEND=Agg python analysis.py`) to only write files.
- Converted from `Dot_Class_refactored.ipynb`; hardcoded absolute paths removed and
  the model code separated from the analysis for reuse.

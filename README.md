# SymBosonKit
A lightweight SymPy 1.14\* add‑on that finally makes symbolic work with **Bosonic ladder operators** painless for quantum‑optics and circuit‑QED researchers. Takes inspiration from the deprecated [SymPsi](https://github.com/sympsi/sympsi/tree/master) library.

---

## Why SymBosonKit?

| What it gives you | How it helps |
|-------------------|--------------|
| **`similarity_transform`** | Performs $(e^{A} B e^{-A})$ while *automatically switching* to closed‑form Lie‑algebra identities when they converge faster than a naïve BCH series. |
| **`hamiltonian_transformation`** | One‑liner to obtain $(UHU^{\dagger} + i \dot U U^{\dagger})$; indispensable for moving to rotating frames, displacements and derived effective models. |
| **`_ad` (multiple‑commutator operator)** | Implements  $(\mathrm{ad}^n_{A}(X)=\underbrace{[A,[A,\dots[A}_{n\ \text{times}},X]\dots]])$ with full operator linearity. |
| **`collect_terms`, `coeff_of`** | Reliable coefficient extraction for *non‑commuting* operators (where `sympy.Expr.coeff` fails). |
| **`drop_terms_containing`, `drop_c_number_terms`** | Fast pruning of rapidly rotating or purely \(c\)-number parts—great for RWA and dispersive limits. |

---

## Quick installation

Copy `symbosonkit.py` into your project folder:

That’s it—no extra dependencies beyond SymPy 1.14*.
Typical workflow — from the accompanying notebook

```python
from sympy import symbols, I, exp, Function
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from symbosonkit import (
    hamiltonian_transformation,           # main workhorse
    similarity_transform,
    collect_terms, coeff_of,
    drop_terms_containing
)

# 1. Define operators and symbols
a      = BosonOp('a'); adag = Dagger(a)
wa, wr, wd, g3, g4, t = symbols('ω_a ω_r ω_d g₃ g₄ t', real=True)
ε      = symbols('ε', complex=True)
ζ      = Function('ζ', complex=True)(t)          # displacement amplitude

# 2. Build a driven Kerr Hamiltonian
H  = wa*adag*a \
   + (g3/2)*(adag**2*a + adag*a**2) \
   + (g4/3)*(adag**3*a + adag*a**3) \
   + ε*(exp(I*wd*t)*adag + exp(-I*wd*t)*a)

# 3. Choose a combined displacement + rotating‑frame unitary
from sympy import exp as e
U = e( ζ*adag - ζ.conjugate()*a ) * e( -I*wd*t*adag*a )

# 4. Transform the Hamiltonian (BCH up to 6th order with Lie‑algebra shortcuts)
H_rot = hamiltonian_transformation(H, U, max_order=6)

# 5. Remove fast‑oscillating pieces (manual RWA)
osc = [exp(I*n*wd*t) for n in range(1,4)] + [exp(-I*n*wd*t) for n in range(1,4)]
H_rwa = drop_terms_containing(H_rot, osc)

# 6. Extract effective Kerr coefficient χ
H_col = collect_terms(H_rwa, adag*a)
χ      = coeff_of(H_col, adag*a)
```

The notebook symbolic_calculations.ipynb walks through this example in depth, including steady‑state solving for ζ(t) and deriving two‑ and three‑photon effective models.


## Roadmap

  - Commuting‑operator tagging for mixed Boson–spin systems
  
  - Time‑dependent operators with automatic derivative handling
  
  - Symbolic master‑equation / Lindbladian utilities
  
  - Liouville‑space similarity transforms
  
  - Pretty‑printing for Jupyter & LaTeX
  
  - Benchmarks to guard against SymPy regressions

## Contributing

Bug reports and pull requests are welcome—open an issue first if you plan a substantial addition so we can coordinate.
## License

MIT

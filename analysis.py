"""
Analysis and figure generation for the two-QD Rabi transport model (§6–§12).

Reproduces the notebook studies: Rabi strong-coupling spectra, deep-strong
transport, Rabi vs Jaynes–Cummings comparison, parameter sweeps, and
Franck–Condon lead-dressing. Figures are written to ./figures/.

Run:  python analysis.py
"""

import os
import copy
from collections import defaultdict

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from numpy.polynomial.laguerre import lagval

import jc_wc
from dot_transport import (
    DotParameters, CavityParameters, LeadParameters,
    PhotonicOperators, ElectronOperators, TunnelOperators, DisplacementOperator,
    HamiltonianBuilder, CavityDecayMatrix, TunnelRateMatrix, RateEquationSolver,
    TransportCalculator,
)

# Directory for saving figures
PLOT_DIR = "figures"
os.makedirs(PLOT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 6.1. Rabi spectrum vs λ
# ----------------------------------------------------------------------
dots = DotParameters(e1=0.6, e2=0.6, t=0.5)
cavity = CavityParameters(n=8, omega=1.0, kappa=1e-3)
M = cavity.n + 1
dim = 4 * M

lam_values = np.linspace(0.0, 3.0, 300)

# Colour the single-charge sector (sector 1) by the hidden parity P=(1<->2)x(a->-a):
# EVEN parity (P=+1) red, ODD parity (P=-1) blue.  Within a parity, same-parity levels
# only avoid-cross, so energy-sorting each parity at every lambda gives smooth, non-crossing
# branches (opposite parities cross freely -> red and blue lines cross, as they should).
# Sectors 0 (empty) and 2 (2e+photons) are flat photonic ladders, kept gray solid / dashed.
phot_par = (-1.0) ** np.arange(M)                       # photon parity (-1)^m

sec0_E = np.empty((len(lam_values), M))
sec2_E = np.empty((len(lam_values), M))
even_E = np.empty((len(lam_values), M))                 # P = +1  (even)
odd_E  = np.empty((len(lam_values), M))                 # P = -1  (odd)

for k, lam in enumerate(lam_values):
    ev, evec, _ = HamiltonianBuilder(cavity, dots, lam).diagonalize()
    sec0_E[k] = np.sort(ev[0:M].real)
    sec2_E[k] = np.sort(ev[3 * M:4 * M].real)
    # parity expectation <psi|P|psi> = 2 Re sum_m (-1)^m conj(L_m) R_m for each sector-1 eigenvector
    L = evec[M:2 * M, M:3 * M]                           # |dot1,m> amplitudes (M x 2M)
    R = evec[2 * M:3 * M, M:3 * M]                       # |dot2,m> amplitudes
    v = 2.0 * np.real(np.sum(phot_par[:, None] * np.conj(L) * R, axis=0))   # length 2M
    order = np.argsort(v)                                # robust M/M split (handles crossings)
    E1 = ev[M:3 * M].real
    odd_E[k]  = np.sort(E1[order[:M]])                   # most negative v -> odd  (P=-1)
    even_E[k] = np.sort(E1[order[M:]])                   # most positive v -> even (P=+1)

EVEN_COLOR, ODD_COLOR, SEC02_COLOR = "tab:red", "tab:blue", "tab:gray"

fig, ax = plt.subplots(figsize=(8, 5))
for j in range(M):                                       # sectors 0 / 2: gray solid / dashed
    ax.plot(lam_values, sec0_E[:, j], color=SEC02_COLOR, ls="-",  lw=0.8, alpha=0.8)
    ax.plot(lam_values, sec2_E[:, j], color=SEC02_COLOR, ls="--", lw=0.8, alpha=0.8)
for j in range(M):                                       # sector 1: parity-coloured
    ax.plot(lam_values, even_E[:, j], color=EVEN_COLOR, ls="-", lw=1.1, alpha=0.95)
    ax.plot(lam_values, odd_E[:,  j], color=ODD_COLOR,  ls="-", lw=1.1, alpha=0.95)

ax.legend(handles=[
    Line2D([0], [0], color=EVEN_COLOR,  ls="-",  label=r"Sector 1, even parity $P=+1$"),
    Line2D([0], [0], color=ODD_COLOR,   ls="-",  label=r"Sector 1, odd parity $P=-1$"),
    Line2D([0], [0], color=SEC02_COLOR, ls="-",  label="Sector 0"),
    Line2D([0], [0], color=SEC02_COLOR, ls="--", label="Sector 2"),
], loc="upper left", fontsize=8)

ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("Eigenvalue")
ax.set_title(rf"Rabi spectrum"
             rf"($e_1=e_2={dots.e1}$, $t={dots.t}$, $\omega={cavity.omega}$, $N={M-1}$)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "rabi_spectrum_all_sectors"+".png"), dpi=150, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------
# 6.1 — parity ladders and the Rabi doublets
# ----------------------------------------------------------------------
# 6.1 — helper: label the single-charge eigenstates by Rabi-partner manifold |n^±>.
#
# The hidden parity P = (1<->2) x (a->-a) splits the single-charge sector into two ladders that
# never mix; same-parity levels only AVOID-cross, so each parity ladder is a set of non-crossing
# branches. Sorting a ladder by energy and pairing consecutive branches recovers the
# Jaynes-Cummings excitation manifolds N: the ladder holding the global ground |g~,0> carries the
# EVEN manifolds N=0,2,4,... (N=0 = the lone ground singlet); the other ladder carries the ODD
# manifolds N=1,3,5,.... Within a manifold, '-' = lower energy, '+' = upper. Because both members
# of a manifold share parity, |n^-> and |n^+> are the GENUINE Rabi partners -- on AND off
# resonance -- even when an opposite-parity level sits between them in energy.
#
# Parity-resolved ordinal scheme: robust at all lambda since there are no true crossings inside a
# parity ladder (so each (N,branch) is one smooth branch). At extreme coupling, where a ladder is
# heavily reshuffled, cross-check against adiabatic continuation from lambda=0.

def label_rabi_partners(eigvals, evec, Ms):
    """Map each single-charge eigenstate (columns Ms..3Ms-1) to its Rabi-partner label.

    Returns {column_index: dict(E, parity=+/-1, N, branch in {'0','-','+','top'},
             label (LaTeX), tag (ascii))}.
    """
    phot_par = np.array([(-1) ** m for m in range(Ms)])            # photon parity (-1)^m
    idx = list(range(Ms, 3 * Ms))                                  # single-charge columns

    def parity(k):                                                 # sign of <psi|P|psi>
        psi = evec[:, k]
        L, R = psi[Ms:2 * Ms], psi[2 * Ms:3 * Ms]                  # |dot1,m>, |dot2,m> amplitudes
        v = 2 * np.real(np.sum(phot_par * np.conj(L) * R))
        return 1.0 if v >= 0 else -1.0

    par = {k: parity(k) for k in idx}
    idx.sort(key=lambda k: eigvals[k].real)                        # by energy
    gp = par[idx[0]]                                               # parity of the ground ladder
    even = [k for k in idx if par[k] == gp]                        # N = 0,2,4,...
    odd  = [k for k in idx if par[k] != gp]                        # N = 1,3,5,...

    def rec(k, p, n, br):
        tag = {'0': '|g~,0>', 'top': f'|{n}(cut)>'}.get(br, f'|{n}{br}>')
        lab = {'0': r'$|\tilde g,0\rangle$',
               'top': rf'$|{n}^{{(cut)}}\rangle$'}.get(br, rf'$|{n}^{br}\rangle$')
        return dict(E=eigvals[k].real, parity=p, N=n, branch=br, label=lab, tag=tag)

    out = {even[0]: rec(even[0], gp, 0, '0')}                      # lone ground singlet
    for ladder, p, n0 in ((even[1:], gp, 2), (odd, -gp, 1)):       # pair up each ladder
        for j in range(0, len(ladder) - 1, 2):
            lo, hi = ladder[j], ladder[j + 1]; n = n0 + (j // 2) * 2
            out[lo] = rec(lo, p, n, '-'); out[hi] = rec(hi, p, n, '+')
        if len(ladder) % 2:                                        # leftover = truncation top singlet
            k = ladder[-1]; out[k] = rec(k, p, n0 + (len(ladder) - 1), 'top')
    return out


# 6.1 — single-charge spectrum, parity-coloured, with explicit Rabi-partner labels |n^±>.
# Branches are grouped into JC manifolds by `label_rabi_partners`: |n^-> (lower) and |n^+> (upper)
# are the genuine Rabi partners -- SAME parity colour, SAME manifold index n -- even though an
# opposite-parity level can sit between them in energy.

dots_s = DotParameters(e1=0.6, e2=0.6, t=0.5)
cav_s  = CavityParameters(n=8, omega=1.0, kappa=1e-3)
Ms = cav_s.n + 1
lam_s = np.linspace(0.0, 5.0, 240)

series = defaultdict(lambda: ([], []))     # (N, branch) -> (lam[], E[])
meta   = {}                                # (N, branch) -> last label record
for lam in lam_s:
    ev, evec, _ = HamiltonianBuilder(cav_s, dots_s, lam).diagonalize()
    for info in label_rabi_partners(ev, evec, Ms).values():
        key = (info['N'], info['branch'])
        series[key][0].append(lam); series[key][1].append(info['E'])
        meta[key] = info

def edge_labels(ax, series, meta, lam, ymax, ymin, nmax=4, gap=0.17):
    """Right-margin |n^±> labels for the low manifolds, vertically dodged with thin leaders."""
    ents = []
    for key, (xs, Es) in series.items():
        info = meta[key]
        if info['branch'] == 'top' or info['N'] > nmax:
            continue
        vis = [i for i, e in enumerate(Es) if ymin < e < ymax - 0.04]    # last in-window point
        if vis:
            i = vis[-1]
            ents.append([xs[i], Es[i], Es[i], info['label'],
                         "tab:red" if info['parity'] > 0 else "tab:blue"])
    ents.sort(key=lambda r: r[2])
    for j in range(1, len(ents)):                                        # greedy upward dodge
        if ents[j][2] - ents[j-1][2] < gap:
            ents[j][2] = ents[j-1][2] + gap
    xL = lam[-1]
    for x0, y0, yl, txt, col in ents:
        ax.annotate(txt, xy=(x0, y0), xytext=(xL + 0.28, yl), textcoords="data",
                    fontsize=8, color=col, va="center", clip_on=False,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5, alpha=0.5),
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.75))

fig, ax = plt.subplots(figsize=(8.8, 6)); ymax, ymin = 4.0, -0.1
for key, (xs, Es) in series.items():
    col = "tab:red" if meta[key]['parity'] > 0 else "tab:blue"
    ax.plot(xs, Es, color=col, lw=1.1)
edge_labels(ax, series, meta, lam_s, ymax, ymin)

ax.legend(handles=[Line2D([0], [0], color="tab:red", lw=2, label=r"$P=+1$  (odd manifolds $N=1,3,\dots$)"),
                   Line2D([0], [0], color="tab:blue", lw=2, label=r"$P=-1$  (even manifolds $N=0,2,\dots$)")],
          loc="lower right", fontsize=9)
ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("single-charge eigenvalue")
ax.set_xlim(lam_s[0], lam_s[-1] + 0.75); ax.set_ylim(ymin, ymax)
ax.set_title("Single-charge sector — Rabi-partner manifolds $|n^\\pm\\rangle$\n"
             "($|n^-\\rangle,|n^+\\rangle$ = same parity colour & manifold $n$; opposite-colour levels cross freely)")
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "rabi_single_charge_spectrum_resonant"+".png"), dpi=150, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------
# 6.1 — off-resonance: the doublets when $\Delta\omega\neq0$
# ----------------------------------------------------------------------
# 6.1 (off-resonance) — single-charge spectrum, parity-coloured, Rabi-partner labels |n^±>.
#   e1=e2=t=1, omega=1:  eps_g=0, eps_e=2, detuning Delta_omega=omega-2t=-1.
# Same manifold labelling (`label_rabi_partners`) as the resonant plot; here NO same-parity pair is
# degenerate at lambda=0, so every |n^->,|n^+> Rabi doublet opens only at finite lambda, finite gap.

dots_o = DotParameters(e1=1, e2=1, t=1)
cav_o  = CavityParameters(n=8, omega=1.0, kappa=1e-3)
Mo = cav_o.n + 1
lam_o = np.linspace(0.0, 5.0, 240)

series = defaultdict(lambda: ([], [])); meta = {}
for lam in lam_o:
    ev, evec, _ = HamiltonianBuilder(cav_o, dots_o, lam).diagonalize()
    for info in label_rabi_partners(ev, evec, Mo).values():
        key = (info['N'], info['branch'])
        series[key][0].append(lam); series[key][1].append(info['E']); meta[key] = info

fig, ax = plt.subplots(figsize=(8.8, 6)); ymax, ymin = 5.0, -0.2
for key, (xs, Es) in series.items():
    col = "tab:red" if meta[key]['parity'] > 0 else "tab:blue"
    ax.plot(xs, Es, color=col, lw=1.1)
edge_labels(ax, series, meta, lam_o, ymax, ymin)        # helper defined in the resonant cell above

################
iref = int(np.argmin(np.abs(lam_o - 1.0))); xref = lam_o[iref]
E1p, E3m = series[(1, '+')][1][iref], series[(3, '-')][1][iref]
ax.plot([xref, xref], [E1p, E3m], color="tab:green", lw=1, alpha=1, linestyle=":", zorder=0.0)
################

# a finite-gap same-parity Rabi doublet (opens only at finite lambda off resonance)
ax.annotate("finite gap (anti-crossing)",
            xy=(1.06, 2.50), xytext=(1.5, 2.50), fontsize=8, color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.25", lw=1.0))
# lambda=0 degeneracy at E=2 is OPPOSITE parity (different manifolds) -> free crossing
ax.annotate(" degeneracy (opposite parity)",
            xy=(0.0, 2.0), xytext=(0.1, 1.30), fontsize=8, color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.25", lw=1.0))

ax.legend(handles=[Line2D([0], [0], color="tab:red", lw=2, label=r"$P=+1$  (odd manifolds)"),
                   Line2D([0], [0], color="tab:blue", lw=2, label=r"$P=-1$  (even manifolds)")],
          loc="lower right", fontsize=9)
ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("single-charge eigenvalue")
ax.set_xlim(lam_o[0], lam_o[-1] + 0.75); ax.set_ylim(ymin, ymax)
ax.set_title("Single-charge sector OFF-RESONANCE ($\\Delta\\omega=\\omega-2t=-1$) — Rabi-partner manifolds $|n^\\pm\\rangle$\n"
             "(every $|n^-\\rangle,|n^+\\rangle$ doublet opens at finite $\\lambda$ with a finite gap)")
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "rabi_single_charge_spectrum_offresonance"+".png"), dpi=150, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------
# 6.1 — spectrum structure: the low-lying polarons in the bare basis
# ----------------------------------------------------------------------
# 6.1 — decomposition of the low-lying single-charge eigenstates (polarons) in the bare basis
# {|g,m>, |e,m>}.  Labels |n^±> come from `label_rabi_partners` (Rabi-partner manifolds), so the two
# members of each doublet carry the SAME n.  At lam=0 each eigenstate is an exact product; at lam=1
# it is a polaron dominated by one bare state plus a small cloud.  (e=0.6, t=0.5, N=10.)

def polaron_decomp(lam, nstate=5, ncomp=5, e=0.6, t=0.5, N=10):
    dots = DotParameters(e1=e, e2=e, t=t)
    cav  = CavityParameters(n=N, omega=1.0, kappa=1e-3)
    ev, evec, _ = HamiltonianBuilder(cav, dots, lam).diagonalize()
    Ms = N + 1
    lab = label_rabi_partners(ev, evec, Ms)
    order = sorted(lab, key=lambda k: ev[k].real)[:nstate]         # single-charge eigenstates by energy
    p0 = evec[:, order[0]]                                          # fix g/e from the ground orbital
    use_minus = abs((p0[Ms:2*Ms] - p0[2*Ms:3*Ms])[0]) >= abs((p0[Ms:2*Ms] + p0[2*Ms:3*Ms])[0])
    for k in order:
        psi = evec[:, k]; L, R = psi[Ms:2*Ms], psi[2*Ms:3*Ms]
        g  = (L - R) / np.sqrt(2) if use_minus else (L + R) / np.sqrt(2)
        eo = (L + R) / np.sqrt(2) if use_minus else (L - R) / np.sqrt(2)
        wg, we = np.abs(g)**2, np.abs(eo)**2
        info = lab[k]; sec = f"P={'+' if info['parity'] > 0 else '-'}1  (N={info['N']})"
        print(f"{info['tag']:>8}  E={ev[k].real:.4f}  {sec}")
        print(f"   {'m':>2} {'|<g,m|.>|^2':>13} {'|<e,m|.>|^2':>13}")
        for m in range(ncomp):
            print(f"   {m:>2} {wg[m]:>13.5f} {we[m]:>13.5f}")
        print()

print("=== lam = 0:  eigenstates are EXACT products ===\n")
polaron_decomp(0.0, nstate=1)
print("=== lam = 1:  low-lying polarons (ground + first two manifolds) ===\n")
polaron_decomp(1.0, nstate=5)


# ----------------------------------------------------------------------
# 6.1 — detuned dots ($e_1\neq e_2$): the parity is broken
# ----------------------------------------------------------------------
# 6.1 — detuned dots (e1 != e2): broken parity.  Single-charge spectrum vs lambda,
# coloured by the CONTINUOUS parity expectation <P> = 2 Re sum_m (-1)^m conj(L_m) R_m
# (no longer +-1 because P is not conserved when e1 != e2).

dots_d = DotParameters(e1=0.6, e2=0.7, t=0.5)        # detuning delta = e1 - e2 = -0.1
cav_d  = CavityParameters(n=8, omega=1.0, kappa=1e-3)
Md     = cav_d.n + 1
lam_d  = np.linspace(0.0, 4.0, 400)
phot_par = (-1.0) ** np.arange(Md)

E_br = np.empty((len(lam_d), 2 * Md))                 # energy-sorted single-charge branches
P_br = np.empty((len(lam_d), 2 * Md))                 # <P> along each branch (continuous)
for k, lam in enumerate(lam_d):
    ev, evec, _ = HamiltonianBuilder(cav_d, dots_d, lam).diagonalize()
    E1 = ev[Md:3 * Md].real
    L = evec[Md:2 * Md, Md:3 * Md]; R = evec[2 * Md:3 * Md, Md:3 * Md]
    Pexp = 2.0 * np.real(np.sum(phot_par[:, None] * np.conj(L) * R, axis=0))   # in [-1, 1]
    order = np.argsort(E1)
    E_br[k] = E1[order]; P_br[k] = Pexp[order]

fig, ax = plt.subplots(figsize=(8.6, 5.8))
norm = plt.Normalize(-1.0, 1.0)
for i in range(2 * Md):                               # one multicoloured line per branch
    pts = np.array([lam_d, E_br[:, i]]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="coolwarm", norm=norm)
    lc.set_array(0.5 * (P_br[:-1, i] + P_br[1:, i]))  # segment colour = mean <P>
    lc.set_linewidth(1.5)
    ax.add_collection(lc)
cb = fig.colorbar(lc, ax=ax, pad=0.02)
cb.set_label(r"$\langle \hat P\rangle$ ")
cb.set_ticks([-1, 0, 1]); cb.set_ticklabels([r"$-1$ (odd)", r"$0$ (mixed)", r"$+1$ (even)"])
ax.set_xlim(lam_d[0], lam_d[-1]); ax.set_ylim(E_br.min() - 0.15, E_br.max() + 0.15)
ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("single-charge eigenvalue")
ax.set_title(rf"Detuned dots $e_1={dots_d.e1},\ e_2={dots_d.e2}$ "
             rf"($\delta={dots_d.e1 - dots_d.e2:+.1f}$): broken parity"
             "\n" r"$\langle\hat P\rangle$ no longer $\pm1$; opposite-parity crossings open avoided gaps")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "rabi_spectrum_detuned_dots"+".png"), dpi=150, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------
# 6.2. Deep strong limit
# ----------------------------------------------------------------------
# 6.2 — deep-strong limit: (a) verify the polaron -> quantum-Rabi mapping is exact;
#       (b) check the large-lambda spectrum is the harmonic ladder  E = e + n*omega.

def _sc_levels(lam, e, t, N, w=1.0):
    """Sorted single-charge eigenvalues (the 2(N+1) levels in columns N+1..3N+2)."""
    ev, _, _ = HamiltonianBuilder(CavityParameters(n=N, omega=w, kappa=1e-3),
                                  DotParameters(e1=e, e2=e, t=t), lam).diagonalize()
    return np.sort(ev[N + 1:3 * (N + 1)].real)

def _rabi_polaron(lam, e, t, N, w=1.0):
    """Lab-frame quantum Rabi model (reverse polaron transform of H)  H~ = e + w a†a + t sz - (w*lam/2)(a+a†) sx + w*lam^2/4."""
    M = N + 1
    a = np.diag(np.sqrt(np.arange(1, M)), 1); ad = a.T; n = ad @ a; I = np.eye(M)
    sz = np.array([[1., 0], [0, -1]]); sx = np.array([[0., 1], [1, 0]]); I2 = np.eye(2)
    H = ((e + w * lam**2 / 4) * np.kron(I, I2) + w * np.kron(n, I2)
         + t * np.kron(I, sz) - (w * lam / 2) * np.kron(a + ad, sx))
    return np.sort(np.linalg.eigvalsh(H))

# (a) exact mapping, lowest 18 levels, Fock space large enough that truncation is irrelevant
print("polaron -> Rabi mapping  (max |Δ| over lowest 18 single-charge levels):")
for lam in (0.5, 1.0, 2.0, 3.0):
    A = _sc_levels(lam, 0.6, 0.5, N=120); B = _rabi_polaron(lam, 0.6, 0.5, N=120)
    print(f"   lambda={lam:>3}:  {np.max(np.abs(A[:18] - B[:18])):.1e}")

# (b) harmonic ladder: pair levels into doublets -> centers, spacings, intra-doublet splitting
    
    '''
    1. E = _sc_levels(...) — gets the sorted single-charge eigenvalues from the physical
      (polaron-frame) dressed-hopping Hamiltonian.
    2. cen (doublet centers) — pairs adjacent levels (E[0],E[1]), (E[2],E[3]), … and averages
      each pair. These are the rung centers, predicted to fall at ε₀ + nω. It keeps the lowest
      ndoub=6 doublets.
    3. spl (intra-doublet splitting) — the gap within each pair, E[2m+1] − E[2m]. This is the
      2·t_eff,n = 2t⟨n|D(λ)|n⟩ = 2t e^{−λ²/2}L_n(λ²) Franck–Condon splitting that goes
      exponentially to zero (part 4).
    4. np.diff(cen) (center spacings) — the rung-to-rung spacing, predicted to approach ω from
      below as λ grows (part 3).
    '''
    
def harmonics(lam, e, t, N=60, ndoub=6, w=1.0):
    E = _sc_levels(lam, e, t, N, w)
    cen = np.array([(E[2*m] + E[2*m+1]) / 2 for m in range(ndoub)])
    spl = np.array([E[2*m+1] - E[2*m]       for m in range(ndoub)])
    return cen, spl, np.diff(cen)

print("\nharmonic ladder at lambda=5  (predicted: center = e + n,  spacing = omega = 1):")
for tag, e, t in (("resonant   (e=0.6, t=0.5)", 0.6, 0.5), ("off-reson. (e=1.0, t=1.0)", 1.0, 1.0)):
    cen, spl, sp = harmonics(5.0, e, t)
    print(f"  {tag}")
    print("     (center-e)/omega : " + " ".join(f"{x-e:6.3f}" for x in cen))
    print("     center spacings  : " + " ".join(f"{x:6.4f}" for x in sp))
    print(f"     rung-0 split = {spl[0]:.2e}   vs  2t e^(-lam^2/2) = {2*t*np.exp(-25/2):.2e}")

print("\ntruncation (resonant, lambda=5) — low-rung spacings are converged already at the plot's n=8:")
for N in (8, 20, 60):
    _, _, sp = harmonics(5.0, 0.6, 0.5, N=N)
    print(f"   n_max={N:>2}: " + " ".join(f"{x:6.4f}" for x in sp))

# figure: (L) spacings -> omega rung by rung;  (C) the exponentially small doublet splitting (log);
#         (R) the same splitting on a linear axis.
lam_grid = np.linspace(0.0, 5.0, 60)
spac_by_rung = [[] for _ in range(5)]; split_by_rung = [[] for _ in range(5)]
for lam in lam_grid:
    _, spl, sp = harmonics(lam, 0.6, 0.5, N=40, ndoub=6)
    for m in range(5):
        spac_by_rung[m].append(sp[m]); split_by_rung[m].append(spl[m])

fig, (axL, axR, axLin) = plt.subplots(1, 3, figsize=(16, 4.2))
for m in range(5):
    axL.plot(lam_grid, spac_by_rung[m], lw=1.3, label=rf"$E_{{{m+1}}}\!-\!E_{{{m}}}$")
    axR.semilogy(lam_grid, np.abs(split_by_rung[m]), lw=1.3, label=f"rung {m}")
    axLin.plot(lam_grid, np.abs(split_by_rung[m]), lw=1.3, label=f"rung {m}")
axL.axhline(1.0, color="k", ls=":", lw=1.2, label=r"$\omega$")
axL.set_xlabel(r"$\lambda$"); axL.set_ylabel("doublet-center spacing")
axL.set_title(r"Approach to harmonicity: level spacings $=\omega$")
axL.set_ylim(0.7, 1.55); axL.legend(fontsize=8, ncol=2, loc="upper right"); axL.grid(alpha=0.3)
axR.semilogy(lam_grid, 2 * 0.5 * np.exp(-lam_grid**2 / 2), "k--", lw=1.6, label=r"$2t_{\mathrm{eff},0}=2t\,e^{-\lambda^2/2}$")
axR.set_xlabel(r"$\lambda$"); axR.set_ylabel(r"doublet splitting $2|t_{\mathrm{eff},n}|$  (log axis)")
axR.set_title(r"Doublet splitting: exponentially small in $\lambda^2$")
axR.set_ylim(1e-7, 3); axR.legend(fontsize=8, ncol=2, loc="lower left"); axR.grid(alpha=0.3, which="both")
axLin.plot(lam_grid, 2 * 0.5 * np.exp(-lam_grid**2 / 2), "k--", lw=1.6, label=r"$2t\,e^{-\lambda^2/2}$")
axLin.set_xlabel(r"$\lambda$"); axLin.set_ylabel(r"doublet splitting $2|t_{\mathrm{eff},n}|$  (linear axis)")
axLin.set_title(r"Doublet splitting: linear axis")
axLin.legend(fontsize=8, ncol=2, loc="upper right"); axLin.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "deep_strong_harmonicity.png"), dpi=150, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------
# **Schematic of the doublet ladder (illustration).** The figure below is a *cartoon*, not a diagonalisation: it
# ----------------------------------------------------------------------
# 6.2 — SCHEMATIC of the doublet ladder (ILLUSTRATION ONLY — not a diagonalization).
# Levels E_0..E_7 vs λ: each rung n is a doublet centred at ε₀+nω (faint dashed line),
# split by 2 t_eff,n = 2t e^{-λ²/2} (t drawn enlarged so the splitting stays visible).
# Shows how the pairs form and how the intra-doublet splitting shrinks while the centres
# stay locked at ε₀+nω.

lam = np.linspace(0.0, 3.0, 300)
e0, w, t = 0.0, 1.0, 0.30                  # schematic numbers (t enlarged for visibility)
half = t * np.exp(-lam**2 / 2)             # half the intra-doublet splitting, ∝ e^{-λ²/2}

fig, ax = plt.subplots(figsize=(7.8, 6.0))
ndoub = 4                                  # rungs n=0..3  ->  levels E_0 .. E_7
for n in range(ndoub):
    c = e0 + n * w
    lo, hi = c - half, c + half
    ax.plot(lam, np.full_like(lam, c), color="0.7", lw=0.9, ls="--", zorder=1)   # doublet centre
    ax.plot(lam, lo, color="C0", lw=2.0, zorder=2)                               # E_{2n}
    ax.plot(lam, hi, color="C3", lw=2.0, zorder=2)                               # E_{2n+1}
    ax.text(-0.10, lo[0], rf"$E_{{{2*n}}}$",   va="center", ha="right", color="C0", fontsize=10)
    ax.text(-0.10, hi[0], rf"$E_{{{2*n+1}}}$", va="center", ha="right", color="C3", fontsize=10)
    ax.text(3.06, c, rf"$\epsilon_0+{n}\,\omega$", va="center", ha="left", color="0.5", fontsize=9)

# annotate ONE intra-doublet splitting (rung n=1) with a double-headed arrow
xm, n_mark = 0.55, 1
cm, hm = e0 + n_mark * w, t * np.exp(-xm**2 / 2)
ax.annotate("", xy=(xm, cm + hm), xytext=(xm, cm - hm),
            arrowprops=dict(arrowstyle="<->", color="k", lw=1.4))
ax.text(xm + 0.07, cm, r"$2t_{\mathrm{eff},n}=2t\,e^{-\lambda^2/2}$", va="center", ha="left", fontsize=9)

# annotate the centre-to-centre spacing = ω (at large λ where the doublets have merged)
xs = 2.6
ax.annotate("", xy=(xs, e0 + 2 * w), xytext=(xs, e0 + 1 * w),
            arrowprops=dict(arrowstyle="<->", color="0.4", lw=1.4))
ax.text(xs - 0.07, e0 + 1.5 * w, r"$\omega$", va="center", ha="right", color="0.4", fontsize=11)

ax.set_xlim(-0.35, 3.6); ax.set_ylim(-0.6, 3.6)
ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("energy (schematic)")
ax.set_title("Schematic: single-charge levels collapse into harmonic doublets "
             r"$\{|g,n\rangle,|e,n\rangle\}$" "\n"
             r"(illustration — splitting enlarged; centres fixed at $\epsilon_0+n\omega$)")
ax.set_yticks([e0 + n for n in range(ndoub)]); ax.grid(alpha=0.2)
plt.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "deep_strong_doublet_schematic.png"), dpi=150, bbox_inches="tight"); plt.show()


# 6.2 — large-λ structure in the LAB frame.  The physical eigenstates become essentially BARE
# molecular–Fock products |g,n>,|e,n> (photons undisplaced, <n>=n), and each {|g,n>,|e,n>} doublet
# splits by the Franck–Condon factor  2t<n|D(λ)|n> = 2t e^{-λ²/2} L_n(λ²).
def _Ln(n, x):
    return lagval(x, [0]*n + [1])

def large_lambda_structure(lam=5.0, e=0.6, t=0.5, N=90, nshow=8):
    ev, evec, _ = HamiltonianBuilder(CavityParameters(n=N, omega=1.0, kappa=1e-3),
                                     DotParameters(e1=e, e2=e, t=t), lam).diagonalize()
    Ms = N + 1
    tags = label_rabi_partners(ev, evec, Ms)
    order = sorted(tags, key=lambda k: ev[k].real)[:nshow]
    p0 = evec[:, order[0]]
    use_minus = abs((p0[Ms:2*Ms] - p0[2*Ms:3*Ms])[0]) >= abs((p0[Ms:2*Ms] + p0[2*Ms:3*Ms])[0])
    print(f"polaron-frame eigenstates at lambda={lam}  (e={e}, t={t}):  bare-product weight and mean photon <n>")
    print(f"  {'state':>7} {'E':>8} {'P':>3} {'<n>':>6}   dominant bare product")
    for k in order:
        psi = evec[:, k]; L, R = psi[Ms:2*Ms], psi[2*Ms:3*Ms]
        g  = (L - R) / np.sqrt(2) if use_minus else (L + R) / np.sqrt(2)
        eo = (L + R) / np.sqrt(2) if use_minus else (L - R) / np.sqrt(2)
        wg, we = np.abs(g)**2, np.abs(eo)**2
        nbar = float(sum(m * (abs(L[m])**2 + abs(R[m])**2) for m in range(Ms)))
        m = int(np.argmax(np.maximum(wg, we)))
        orb, w = ('g', wg[m]) if wg[m] >= we[m] else ('e', we[m])
        print(f"  {tags[k]['tag']:>7} {ev[k].real:8.4f} {int(tags[k]['parity']):>+3} {nbar:6.3f}   {w:.3f} |{orb},{m}>")
    E = np.sort(ev[Ms:3*Ms].real)
    print("\n  {|g,n>,|e,n>} doublet splitting   vs   2t e^(-lam^2/2)|L_n(lam^2)| = 2t|<n|D(lam)|n>|:")
    for n in range(4):
        meas = E[2*n+1] - E[2*n]; fc = 2*t*np.exp(-lam**2/2)*abs(_Ln(n, lam**2))
        print(f"    rung n={n}:   split = {meas:.3e}    FC = {fc:.3e}")

large_lambda_structure(lam=5.0)


# ----------------------------------------------------------------------
# 7. Deep strong transport
# ----------------------------------------------------------------------
# 7 — Deep strong transport: I-V at lambda >> 1.  Four scenarios isolating the roles of the
# inter-dot dressing (lam) and the lead dressings (lam_L, lam_R).  deep_strong_iv IS the
# sequential-tunneling pipeline of iv_curve_sweep (sec 10), inlined; deep_strong_iv_fc is the
# same with INDEPENDENT lead displacements.  Only the sec 3-5 classes are used.

def deep_strong_iv(dots, cavity, leads_template, lam, V_values, *, dress_leads):
    """One-sided-bias (mu_L=V, mu_R=0) Rabi I(V).  dress_leads=True sets lam_L=lam_R=lam."""
    eigvals, eigvecs, _ = HamiltonianBuilder(cavity, dots, lam).diagonalize()
    Gamma_ph = CavityDecayMatrix(cavity, eigvals, eigvecs).build()
    transp = TransportCalculator(cavity, leads_template)
    I = np.empty(len(V_values))
    for j, V in enumerate(V_values):
        leads = copy.deepcopy(leads_template)
        leads.mu_L, leads.mu_R = V, 0.0
        if dress_leads:
            leads.lam_L = leads.lam_R = lam
        G_L, _G_R, G_tot = TunnelRateMatrix(cavity, dots, leads, eigvals, eigvecs).build()
        P = RateEquationSolver(G_tot, Gamma_ph, cavity).steady_solver()
        I[j] = transp.compute_current(G_L, P)
    return I


def deep_strong_iv_fc(dots, cavity, leads_template, lam, V_values, *, lam_L, lam_R):
    """As deep_strong_iv but with INDEPENDENT lead displacements (lam = inter-dot dressing)."""
    eigvals, eigvecs, _ = HamiltonianBuilder(cavity, dots, lam).diagonalize()
    Gamma_ph = CavityDecayMatrix(cavity, eigvals, eigvecs).build()
    transp = TransportCalculator(cavity, leads_template)
    I = np.empty(len(V_values))
    for j, V in enumerate(V_values):
        leads = copy.deepcopy(leads_template)
        leads.mu_L, leads.mu_R = V, 0.0
        leads.lam_L, leads.lam_R = lam_L, lam_R
        G_L, _G_R, G_tot = TunnelRateMatrix(cavity, dots, leads, eigvals, eigvecs).build()
        P = RateEquationSolver(G_tot, Gamma_ph, cavity).steady_solver()
        I[j] = transp.compute_current(G_L, P)
    return I


def _Nds(lam):
    return max(8, int(np.ceil(3 * lam**2)) + 10)        # Fock cutoff ~ 3*lam^2 + buffer

# --- parameters: resonant bright-bonding dot (eps_g = e - t = 0.1 at lambda=0), kappa = 1 ---
omega_ds = 1.0
dots_ds  = DotParameters(e1=0.6, e2=0.6, t=0.5)
leads_ds = LeadParameters(gamma_L=1e-3, gamma_R=1e-3, mu_L=0.0, mu_R=0.0, mu_0=0.0,
                          T_L=1e-3, T_R=1e-3, lam_L=0.0, lam_R=0.0)
lam_ds   = [1.0, 2.0, 3.0]
V_ds     = np.linspace(-0.3, 5.0, 240)
kappa_ds = 1.0

def _cav(*lams):
    return CavityParameters(n=_Nds(max(lams)), omega=omega_ds, kappa=kappa_ds)   # cutoff from the largest displacement

# lambda = 0 non-interacting reference (lam = lam_L = lam_R = 0)
I_ref = deep_strong_iv(dots_ds, CavityParameters(n=8, omega=omega_ds, kappa=kappa_ds),
                       leads_ds, 0.0, V_ds, dress_leads=False)

cmap = plt.cm.viridis(np.linspace(0.15, 0.82, len(lam_ds)))

# --- panel data ---
bare      = {lam: deep_strong_iv(dots_ds, _cav(lam), leads_ds, lam, V_ds, dress_leads=False)
             for lam in lam_ds}
dressed   = {lam: deep_strong_iv(dots_ds, _cav(lam), leads_ds, lam, V_ds, dress_leads=True)
             for lam in lam_ds}
# scenario 3: leads fixed (lam_L=lam_R=1), inter-dot lam = 2, 3  (merged into dressed panel)
scen3     = {lam: deep_strong_iv_fc(dots_ds, _cav(lam, 1.0), leads_ds, lam, V_ds, lam_L=1.0, lam_R=1.0)
             for lam in (2.0, 3.0)}
# scenario 1: only LEFT lead dressed (lam = lam_R = 0), sweep lam_L
leftonly  = {k: deep_strong_iv_fc(dots_ds, _cav(k), leads_ds, 0.0, V_ds, lam_L=k, lam_R=0.0)
             for k in (1.0, 2.0, 3.0)}
# scenario 2: only RIGHT lead dressed (lam = lam_L = 0), sweep lam_R
rightonly = {k: deep_strong_iv_fc(dots_ds, _cav(k), leads_ds, 0.0, V_ds, lam_L=0.0, lam_R=k)
             for k in (1.0, 2.0, 3.0)}

# --- figure: 2x2 ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
(axB, axD), (axL, axR) = axes
def _ref(ax): ax.plot(V_ds, I_ref, color="0.6", ls="--", lw=1.2, label=r"$\lambda=0$ (non-int.)")

_ref(axB)
for lam, col in zip(lam_ds, cmap):
    axB.plot(V_ds, bare[lam], color=col, lw=1.7, label=rf"$\lambda={lam:g}$")
axB.set_title(r"Bare leads  ($\lambda_L=\lambda_R=0$, sweep $\lambda$)")

_ref(axD)
for lam, col in zip(lam_ds, cmap):
    axD.plot(V_ds, dressed[lam], color=col, lw=1.7, label=rf"$\lambda_L=\lambda_R=\lambda={lam:g}$")
for lam, col in zip((2.0, 3.0), ("tab:brown", "tab:pink")):
    axD.plot(V_ds, scen3[lam], color=col, lw=1.7, ls="--",
             label=rf"$\lambda_L=\lambda_R=1,\ \lambda={lam:g}$")
axD.set_title(r"Dressed leads ($\lambda_L=\lambda_R=\lambda$) + fixed leads ($\lambda_L=\lambda_R=1$)")

_ref(axL)
for k, col in zip((1.0, 2.0, 3.0), cmap):
    axL.plot(V_ds, leftonly[k], color=col, lw=1.7, label=rf"$\lambda_L={k:g}$")
axL.set_title(r"Left lead only  ($\lambda=\lambda_R=0$, sweep $\lambda_L$)")

_ref(axR)
for k, col in zip((1.0, 2.0, 3.0), cmap):
    axR.plot(V_ds, rightonly[k], color=col, lw=1.7, label=rf"$\lambda_R={k:g}$")
axR.set_title(r"Right lead only  ($\lambda=\lambda_L=0$, sweep $\lambda_R$)")

for ax in axes.flat:
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="center right")
for ax in (axL, axR): ax.set_xlabel(r"$V_b$")
for ax in (axB, axL): ax.set_ylabel(r"current $I$")
fig.suptitle(r"Deep-strong transport: I–V at $\kappa=1$, resonant ($e=0.6,\ t=0.5,\ \omega=1$); "
             r"roles of inter-dot $\lambda$ vs lead $\lambda_L,\lambda_R$", y=1.005)
fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "deep_strong_iv_grid.png"), dpi=150, bbox_inches="tight"); plt.show()


# 7 — Fock-cutoff convergence: the in-window I-V steps must not move with N.
# Recompute the most demanding case (lambda=3) at two cutoffs; bare and dressed.
V_chk = np.linspace(-0.3, 5.0, 130)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for N_chk, ls in ((_Nds(3.0), "-"), (_Nds(3.0) + 16, "--")):
    cav = CavityParameters(n=N_chk, omega=omega_ds, kappa=kappa_ds)
    for dress, col in ((False, "tab:blue"), (True, "tab:red")):
        I = deep_strong_iv(dots_ds, cav, leads_ds, 3.0, V_chk, dress_leads=dress)
        ax.plot(V_chk, I, ls=ls, color=col, lw=1.5,
                label=f"N={N_chk}, {'dressed' if dress else 'bare'}")
ax.set_xlabel(r"$V_b$"); ax.set_ylabel(r"$I$"); ax.grid(alpha=0.3)
ax.set_title(rf"Cutoff convergence at $\lambda=3$ "
             rf"($N={_Nds(3.0)}$ solid vs $N={_Nds(3.0)+16}$ dashed)")
ax.legend(fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "deep_strong_iv_convergence.png"), dpi=150, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------
# 7 — note: persistent current and the secular limit
# ----------------------------------------------------------------------
# 7 — diagnostic: bare first-plateau current pushed into the deep-strong regime (lambda up to 5).
# The parity-protected molecular doublet stays delocalised, so the SECULAR rate equation keeps the
# current lead-limited (~1) even after the gap 2 t_eff drops below gamma at lambda_c (where 2 t_eff = gamma).
# A non-secular treatment would instead give I ~ t_eff^2/gamma there. Single plateau point per lambda
# (V well above the e-onset), reusing the sec 7 parameters (dots_ds, leads_ds, kappa_ds, _Nds).

def _bare_plateau(lam, V=2.0):
    cav = CavityParameters(n=_Nds(lam), omega=omega_ds, kappa=kappa_ds)
    leads = copy.deepcopy(leads_ds); leads.mu_L, leads.mu_R = V, 0.0
    ev, evec, _ = HamiltonianBuilder(cav, dots_ds, lam).diagonalize()
    Gph = CavityDecayMatrix(cav, ev, evec).build()
    G_L, _, G_tot = TunnelRateMatrix(cav, dots_ds, leads, ev, evec).build()
    P = RateEquationSolver(G_tot, Gph, cav).steady_solver()
    return TransportCalculator(cav, leads).compute_current(G_L, P)

lam_ext = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
I_ext   = np.array([_bare_plateau(l) for l in lam_ext])
t_ds, gamma = dots_ds.t, leads_ds.gamma_L
gap_ratio = 2 * t_ds * np.exp(-lam_ext**2 / 2) / gamma      # 2 t_eff / gamma
lam_c = np.sqrt(2 * np.log(2 * t_ds / gamma))               # 2 t_eff = gamma

fig, axL = plt.subplots(figsize=(8, 4.8))
axL.plot(lam_ext, I_ext, "o-", color="tab:blue", lw=1.9, ms=6,
         label=r"bare plateau $I$ (secular RE)")
axL.axhline(1.0, color="0.75", ls=":", lw=1.0)
axL.set_xlabel(r"$\lambda$"); axL.set_ylabel(r"plateau current $I$", color="tab:blue")
axL.tick_params(axis="y", labelcolor="tab:blue"); axL.set_ylim(0, 1.12)
axR = axL.twinx()
axR.semilogy(lam_ext, gap_ratio, "s--", color="tab:red", lw=1.6, ms=5,
             label=r"$2t_{\mathrm{eff}}/\gamma$")
axR.axhline(1.0, color="tab:red", ls=":", lw=1.0)
axR.set_ylabel(r"$2t_{\mathrm{eff}}/\gamma$  (log)", color="tab:red")
axR.tick_params(axis="y", labelcolor="tab:red")
axL.axvline(lam_c, color="k", ls="-.", lw=1.3)
axL.text(lam_c + 0.06, 0.30,
         rf"$2t_{{\mathrm{{eff}}}}=\gamma$" "\n" rf"$\lambda_c\approx{lam_c:.2f}$"
         "\n(secular RE breaks\n down beyond here)", fontsize=8.5, va="top")
axL.set_title(r"Persistent current vs the gap-to-$\gamma$ crossover (bare leads, $\kappa=1$)")
h1, l1 = axL.get_legend_handles_labels(); h2, l2 = axR.get_legend_handles_labels()
axL.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "deep_strong_persistent_current.png"), dpi=150, bbox_inches="tight"); plt.show()


# 7 — bare-leads I-V deep into the strong-coupling regime (lambda = 3, 4, 5).
# Full I-V companion to the persistent-current diagnostic: the curves collapse onto a
# SINGLE step at the bare onsite e = 0.6 (LP/UP merged, eps_g -> e) and still reach full
# current -- the molecular doublet stays delocalised (secular RE; see the note above).
# Reuses deep_strong_iv and the sec 7 parameters; cutoff N = 3 lambda^2 + 10 per curve.
V_hi   = np.linspace(-0.3, 2.2, 130)
lam_hi = [3.0, 4.0, 5.0]
I_ref_hi = deep_strong_iv(dots_ds, CavityParameters(n=8, omega=omega_ds, kappa=kappa_ds),
                          leads_ds, 0.0, V_hi, dress_leads=False)
cmap_hi = plt.cm.plasma(np.linspace(0.15, 0.72, len(lam_hi)))

fig, ax = plt.subplots(figsize=(7.6, 4.8))
ax.plot(V_hi, I_ref_hi, color="0.6", ls="--", lw=1.3, label=r"$\lambda=0$ (non-int.: $e\mp t$)")
for lam, col in zip(lam_hi, cmap_hi):
    cav = CavityParameters(n=_Nds(lam), omega=omega_ds, kappa=kappa_ds)
    I = deep_strong_iv(dots_ds, cav, leads_ds, lam, V_hi, dress_leads=False)
    ax.plot(V_hi, I, color=col, lw=1.9, label=rf"$\lambda={lam:g}\ (N={_Nds(lam)})$")
ax.axvline(0.6, color="k", ls=":", lw=1.1); ax.text(0.62, 0.08, r"$e=0.6$", fontsize=9)
ax.set_xlabel(r"$V_b$"); ax.set_ylabel(r"current $I$"); ax.grid(alpha=0.3)
ax.set_ylim(-0.03, 1.12)
ax.set_title(r"Bare-leads I–V, deep strong ($\lambda=3,4,5$): single step at the bare onsite $e$")
ax.legend(fontsize=8.5, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "deep_strong_bare_hi_lambda.png"), dpi=150, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------
# 8. Rabi vs Jaynes–Cummings spectrum comparison
# ----------------------------------------------------------------------
# JCBuilder copied verbatim from weak_coupling.ipynb (Cell 0) so this
# cell stays self-contained and doesn't need `import_ipynb`.

class JCBuilder:
    def __init__(self, cavity: CavityParameters, dots: DotParameters, lam):
        self.cavity = cavity     # store the whole cavity object
        self.dots = dots
        self.lam = lam
        
        # phonon cutoff
        self.n = self.cavity.n
        self.M = self.n + 1
        self.dim = 4 * self.M
        
        # Build the full Hamiltonian once and store it
        self.H = self.build()
        
        self.H0 = self._build_sector_0()
        self.H1 = self._build_sector_1()
        self.H2 = self._build_sector_2()
    
    """Now let's build the Hamiltonian step-by-step."""
    # -----------------------------
    # Sector builders
    # -----------------------------

    def _build_sector_0(self):
        """Constructs the zero-particle sector Hamiltonian."""
        H0 = np.zeros((self.M, self.M), dtype=np.float64)
        for n in range(self.M):
            H0[n, n] = self.cavity.omega * n
        return H0

    def _build_sector_2(self):
        """Constructs the two-particle sector Hamiltonian."""
        H2 = np.zeros((self.M, self.M), dtype=np.float64)
        for n in range(self.M):
            H2[n, n] = self.dots.e1 + self.dots.e2 + self.cavity.omega * n
        return H2

    def _build_sector_1(self):
        """Constructs the single-particle sector Hamiltonian in the basis:
          n=photon number  ===> basis = {|g,0>; |e,0>,|g,1>; |e,1>,|g,2>; ...; |e,n-1>, |g,n>; ... |e,n>}
        """
        eg = (self.dots.e1 + self.dots.e2)/2 - np.sqrt((self.dots.e1 - self.dots.e2)**2/4 + (self.dots.t)**2)
        ee = (self.dots.e1 + self.dots.e2)/2 + np.sqrt((self.dots.e1 - self.dots.e2)**2/4 + (self.dots.t)**2)
        
        H1 = np.zeros((2 * self.M, 2 * self.M), dtype=np.float64)

        # diagonal  block
        for n in range(self.M):
            H1[2*n, 2*n] = eg + self.cavity.omega * n
            H1[2*n+1, 2*n+1] = ee + self.cavity.omega * n

        # Hopping
        for m in range(self.n):
            r = 2*m + 1
            c = r + 1
            H1[r, c] = np.sqrt(m+1)*self.lam
            H1[c, r] = np.sqrt(m+1)*self.lam

        return (H1 + H1.conj().T) / 2.0

    # -----------------------------
    # Full Hamiltonian
    # -----------------------------

    def build(self):
        H0 = self._build_sector_0()
        H1 = self._build_sector_1()
        H2 = self._build_sector_2()

        H = np.zeros((self.dim, self.dim), dtype=np.float64)

        idx0 = list(range(self.M))
        idx1 = list(range(self.M, 3*self.M))
        idx2 = list(range(3*self.M, 4*self.M))

        # Fill blocks
        H[np.ix_(idx0, idx0)] = H0
        H[np.ix_(idx1, idx1)] = H1
        H[np.ix_(idx2, idx2)] = H2

        return (H + H.conj().T) / 2.0

    # -----------------------------
    # Diagonalization
    # -----------------------------

    def diagonalize(self):
    
        M = self.M
        dim = self.dim
    
        # --- eigenvalues of diagonal sectors ---
        eigvals_0 = np.diag(self.H0)
        eigvals_2 = np.diag(self.H2)
    
        # eigenvectors are identity
        eigvecs_0 = np.eye(M)
        eigvecs_2 = np.eye(M)
    
        # --- diagonalize JC sector ---
        eigvals_1, eigvecs_1 = eigh(self.H1)
    
        # --- embed eigenvectors into full Hilbert space ---
        V0 = np.zeros((dim, M), dtype=np.complex128)
        V1 = np.zeros((dim, 2*M), dtype=np.complex128)
        V2 = np.zeros((dim, M), dtype=np.complex128)
    
        V0[0:M, :] = eigvecs_0
        V1[M:3*M, :] = eigvecs_1
        V2[3*M:4*M, :] = eigvecs_2
    
        # --- merge spectra ---
        eigvals = np.concatenate([eigvals_0, eigvals_1, eigvals_2])
        eigvecs = np.hstack([V0, V1, V2])
    
        # --- charge sector labels ---
        state_charge = (
            [0]*M +
            [1]*(2*M) +
            [2]*M
        )
    
        return eigvals, eigvecs, state_charge

    def diagonalize_ordered(self):

        M = self.M
        dim = self.dim
        
        # --- eigenvalues of diagonal sectors ---
        eigvals_0 = np.diag(self.H0)
        eigvals_2 = np.diag(self.H2)
        
        eigvecs_0 = np.eye(M)
        eigvecs_2 = np.eye(M)
        
        # --- diagonalize JC sector ---
        eigvals_1, eigvecs_1 = eigh(self.H1)
        
        # --- reorder eigenstates to match original basis ---
        overlaps = np.abs(eigvecs_1)**2
        dominant_idx = np.argmax(overlaps, axis=0)
        order = np.argsort(dominant_idx)
        
        eigvals_1 = eigvals_1[order]
        eigvecs_1 = eigvecs_1[:, order]
        
        # --- embed eigenvectors ---
        V0 = np.zeros((dim, M), dtype=np.complex128)
        V1 = np.zeros((dim, 2*M), dtype=np.complex128)
        V2 = np.zeros((dim, M), dtype=np.complex128)
        
        V0[0:M, :] = eigvecs_0
        V1[M:3*M, :] = eigvecs_1
        V2[3*M:4*M, :] = eigvecs_2
        
        eigvals_ordered = np.concatenate([eigvals_0, eigvals_1, eigvals_2])
        eigvecs_ordered = np.hstack([V0, V1, V2])
        
        state_charge_ordered = (
            [0]*M +
            [1]*(2*M) +
            [2]*M
        )
        
        return eigvals_ordered, eigvecs_ordered, state_charge_ordered
            
#################
##### PRIME 
#################

# ----------------------------------------------------------
# Sweep lambda, collect sector-1 spectrum from both models
# ----------------------------------------------------------
dots = DotParameters(e1=0.6, e2=0.6, t=0.5)
cavity = CavityParameters(n=5, omega=1.0, kappa=1e-3)

M = cavity.n + 1              # photon-block size
lam_values = np.linspace(0.0, 0.5, 200)

spectra_Rabi = np.empty((len(lam_values), 2 * M))
spectra_JC = np.empty((len(lam_values), 2 * M))

for k, lam in enumerate(lam_values):
    # Rabi (full counter-rotating terms)
    rabi = HamiltonianBuilder(cavity, dots, lam)
    evals_R, *_ = rabi.diagonalize()
    spectra_Rabi[k] = evals_R[M:3 * M]

    # Jaynes-Cummings (rotating-wave approximation).
    # The polaron expansion gives g_JC = t * lam_Rabi, so feed the JC
    # builder dots.t * lam to share a common physical coupling axis.
    
    jc = JCBuilder(cavity, dots, dots.t * lam)
    evals_J, *_ = jc.diagonalize_ordered()
    spectra_JC[k] = evals_J[M:3 * M]

# -------------------------------------------------------------------
# Plot: Rabi as dashed lines, JC as markers; matched colors per level
# -------------------------------------------------------------------
n_levels = spectra_Rabi.shape[1]
colors = plt.cm.viridis(np.linspace(0.0, 0.95, n_levels))

fig, ax = plt.subplots(figsize=(8, 5))
for i in range(n_levels):
    ax.plot(lam_values, spectra_Rabi[:, i],
            linestyle="-", lw=1.0, color=colors[i], alpha=0.9)
    ax.plot(lam_values, spectra_JC[:, i],
            linestyle="-.", marker="o", markersize=2.5,
            markevery=8, color=colors[i], alpha=0.9)

# Legend handles: one entry per model, color-agnostic
legend_handles = [
    Line2D([0], [0], linestyle="-", color="black",
           label="Rabi (strong coupling)"),
    Line2D([0], [0], linestyle="-.", marker="o", markersize=4,
           color="black", label="JC (RWA)"),
]

ax.legend(handles=legend_handles, loc="upper left", frameon=True)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"Eigenvalue (sector 1)")
ax.set_title(rf"Rabi vs JC spectrum (JC at $g=t\lambda$)  "
             rf"($e_1=e_2={dots.e1}$, $t={dots.t}$, "
             rf"$\omega={cavity.omega}$, $N={M-1}$)")

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "rabi_vs_jc_spectrum"+".png"), dpi=150, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------
# 9. Rabi vs Jaynes–Cummings I–V comparison
# ----------------------------------------------------------------------
def rabi_iv(dots, cavity, leads_template, lam, V_values):
    """One-sided-bias (mu_L=V, mu_R=0) I(V) for the Rabi model. Hamiltonian +
    photon decay built once; only the Fermi-dressed tunnel rates rebuilt per V."""
    leads = copy.deepcopy(leads_template)
    eigvals, eigvecs, _ = HamiltonianBuilder(cavity, dots, lam).diagonalize()
    Gamma_ph = CavityDecayMatrix(cavity, eigvals, eigvecs).build()
    transp = TransportCalculator(cavity, leads)
    I = np.empty(len(V_values))
    for j, V in enumerate(V_values):
        leads.mu_L, leads.mu_R = V, 0.0
        G_L, _G_R, G_tot = TunnelRateMatrix(cavity, dots, leads, eigvals, eigvecs).build()
        P = RateEquationSolver(G_tot, Gamma_ph, cavity).steady_solver()
        I[j] = transp.compute_current(G_L, P)
    return I


def rabi_thresholds(t, lam, strength_cut=0.02):
    """Per-panel Rabi step locations (raw energies, |0,0> at 0):
    (eps_g, [polariton energies], step_20). Polaritons = up to 2 brightest
    sector-1 states reachable from |0,0> besides the ground."""
    e = eps_g + t
    M = n_cut + 1
    dots = DotParameters(e1=e, e2=e, t=t)
    cav = CavityParameters(n=n_cut, omega=omega, kappa=1e-12)
    ev, evec, _ = HamiltonianBuilder(cav, dots, lam).diagonalize()
    probe = LeadParameters(gamma_L=1.0, gamma_R=1e-9, mu_L=50.0, mu_R=0.0, mu_0=0.0,
                           T_L=1e-3, T_R=1e-3, lam_L=0.0, lam_R=0.0)
    G_L, _G_R, _G_tot = TunnelRateMatrix(cav, dots, probe, ev, evec).build()
    i00 = int(np.argmin(np.abs(ev[:M])))
    s1 = list(range(M, 3 * M))
    energies = np.array([ev[f] for f in s1])
    strengths = np.array([G_L[i00, f] for f in s1])      # Dot_Class: Gamma[i,f]=rate i->f
    e_ground = energies.min()
    bright = [(en, s) for en, s in zip(energies, strengths)
              if en > e_ground + 1e-6 and s > strength_cut]
    bright.sort(key=lambda x: -x[1])
    pol = sorted(en for en, _ in bright[:2])
    return e_ground, pol, 2 * e - e_ground


# ---- comparison grid parameters ----
omega = 1.0
eps_g = 0.1                              # bonding ground kept in transport window
t_values = [0.4, 0.5, 0.6]              # Delta_omega = omega - 2t = +0.2, 0, -0.2
detuning_titles = [rf"$\Delta\omega={omega - 2*t:+.1f}$ ($t={t}$)" for t in t_values]
lam_values = [0.0, 0.2, 0.5]            # polaron lambda (Rabi); JC fed t*lambda
grid_kappas = [0.0, 1.0]                # the two fixed-kappa grids (sec 9a, 9b)
n_cut = 6
V = np.linspace(-0.5, 4.0, 800)         # step width ~ T (~0.004); raise N for sharper steps


def _kappa(k):
    return k if k > 0 else 1e-12        # strict 0 -> degenerate steady state


leads_rabi = LeadParameters(gamma_L=1e-3, gamma_R=1e-3, mu_L=0.0, mu_R=0.0,
                            mu_0=0.0, T_L=1e-3, T_R=1e-3, lam_L=0.0, lam_R=0.0)
leads_jc = jc_wc.LeadParameters(gamma_L=1e-3, gamma_R=1e-3, mu_L=0.0, mu_R=0.0,
                                mu_0=0.0, T_L=1e-3, T_R=1e-3, lambda_L=0.0, lambda_R=0.0)

# ---- compute every curve once: IV[(t, lam, kappa, model)] = I(V) ----
IV = {}
for t in t_values:
    e = eps_g + t
    dots = DotParameters(e1=e, e2=e, t=t)
    for lam in lam_values:
        for kappa in grid_kappas:
            cav_r = CavityParameters(n=n_cut, omega=omega, kappa=_kappa(kappa))
            cav_j = jc_wc.CavityParameters(n=n_cut, omega=omega, kappa=_kappa(kappa))
            IV[(t, lam, kappa, "Rabi")] = rabi_iv(dots, cav_r, leads_rabi, lam, V)
            IV[(t, lam, kappa, "JC")] = jc_wc.iv_curve(dots, cav_j, leads_jc, t * lam, V)
print(f"Computed {len(IV)} I-V curves over kappa in {grid_kappas}.")


def plot_iv_grid(kappa_plot):
    """Rabi (solid black) vs JC (dashed orange) I-V at a single fixed kappa.
    Rows = lam_values, columns = detunings. Per-panel Rabi thresholds (dotted
    vlines: eps_g/LP/UP/|2,0> step) and LP/UP zoom insets for lam in {0.2, 0.5}."""
    rabi_col, jc_col = "black", "tab:orange"
    fig, axes = plt.subplots(len(lam_values), len(t_values),
                             figsize=(15, 3.6 * len(lam_values)),
                             sharex=True, sharey=True, squeeze=False)
    for r, lam in enumerate(lam_values):
        for c, t in enumerate(t_values):
            ax = axes[r, c]
            ax.plot(V, IV[(t, lam, kappa_plot, "Rabi")], color=rabi_col, ls="-",  lw=1.3)
            ax.plot(V, IV[(t, lam, kappa_plot, "JC")],   color=jc_col,   ls="--", lw=1.3)
            e_g, pol, step20 = rabi_thresholds(t, lam)
            ax.axvline(e_g, color="0.4", ls=":", lw=1.0)
            for pe, pcol in zip(pol, ["tab:blue", "tab:red"]):
                ax.axvline(pe, color=pcol, ls=":", lw=1.0)
            ax.axvline(step20, color="tab:green", ls=":", lw=1.0)
            # second electron from the lower polariton: |LP> -> |2,1> at 2e + omega - E_LP.
            # Only a kappa->0 feature (the 1-photon |2,1> relaxes at finite kappa), so
            # the marker is drawn only on the kappa=0 grid.
            if len(pol) == 2 and np.isclose(kappa_plot, 0.0):
                lp21 = 2 * (eps_g + t) + omega - pol[0]
                ax.axvline(lp21, color="tab:purple", ls=":", lw=1.0)
            ax.grid(True, alpha=0.3)
            # LP<->UP zoom inset on a fine local grid (lambda in {0.2, 0.5})
            if any(np.isclose(lam, x) for x in (0.2, 0.5)) and len(pol) == 2:
                elp, eup = pol
                margin = 0.25 * (eup - elp)
                xlo, xhi = elp - margin, eup + margin
                axin = ax.inset_axes([0.62, 0.10, 0.34, 0.34])
                Vz = np.linspace(xlo, xhi, 400)
                e_dot = eps_g + t
                dots_z = DotParameters(e1=e_dot, e2=e_dot, t=t)
                cav_r = CavityParameters(n=n_cut, omega=omega, kappa=_kappa(kappa_plot))
                cav_j = jc_wc.CavityParameters(n=n_cut, omega=omega, kappa=_kappa(kappa_plot))
                Ir = rabi_iv(dots_z, cav_r, leads_rabi, lam, Vz)
                Ij = jc_wc.iv_curve(dots_z, cav_j, leads_jc, t * lam, Vz)
                axin.plot(Vz, Ir, color=rabi_col, ls="-",  lw=1.0)
                axin.plot(Vz, Ij, color=jc_col,   ls="--", lw=1.0)
                axin.axvline(elp, color="tab:blue", ls=":", lw=0.9)
                axin.axvline(eup, color="tab:red",  ls=":", lw=0.9)
                if np.isclose(kappa_plot, 0.0):
                    axin.axvline(2 * (eps_g + t) + omega - elp, color="tab:purple", ls=":", lw=0.9)
                yz = np.concatenate([Ir, Ij])
                axin.set_xlim(xlo, xhi)
                axin.set_ylim(yz.min() - 0.02, yz.max() + 0.02)
                axin.tick_params(labelsize=6)
                axin.set_title("LP/UP zoom", fontsize=6)
                v_mid = 0.5 * (elp + eup)
                i_mid = float(np.interp(v_mid, Vz, Ir))
                ax.annotate("", xy=(v_mid, i_mid), xycoords="data",
                            xytext=(0.62, 0.46), textcoords="axes fraction",
                            arrowprops=dict(arrowstyle="->", color="0.25", lw=1.1))
            if r == 0:
                ax.set_title(detuning_titles[c])
            if c == 0:
                ax.set_ylabel(rf"$\lambda={lam:g}$" + "\n\n" + r"$I$")
            if r == len(lam_values) - 1:
                ax.set_xlabel(r"$V_b$")
    curve_handles = [Line2D([0], [0], color=rabi_col, ls="-",  label="Rabi"),
                     Line2D([0], [0], color=jc_col,   ls="--", label=r"JC ($g=t\lambda$)")]
    thr_handles = [Line2D([0], [0], color="0.4",      ls=":", label=r"$\varepsilon_g$"),
                   Line2D([0], [0], color="tab:blue", ls=":", label="LP"),
                   Line2D([0], [0], color="tab:red",  ls=":", label="UP"),
                   Line2D([0], [0], color="tab:green", ls=":", label=r"$|2,0\rangle$ step")]
    if np.isclose(kappa_plot, 0.0):
        thr_handles.append(
            Line2D([0], [0], color="tab:purple", ls=":", label=r"$|LP\rangle\to|2,1\rangle$"))
    lead_ax = axes[0, -1]
    leg_main = lead_ax.legend(handles=curve_handles, loc="upper right", fontsize=8)
    lead_ax.add_artist(leg_main)
    lead_ax.legend(handles=thr_handles, loc="lower right", fontsize=7,
                   title="threshold $V_b$", title_fontsize=7)
    fig.suptitle(rf"Rabi vs JC I–V at fixed $\kappa={kappa_plot:g}$" 
                 "\n" 
                 r"(rows = $\lambda$, columns = detuning)", y=1.002)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"rabi_jc_iv_grid_kappa{kappa_plot:g}"+".png"), dpi=150, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------
# 9a. Fixed κ = 0 — rows = λ, columns = detuning
# ----------------------------------------------------------------------
plot_iv_grid(0.0)


# ----------------------------------------------------------------------
# 9b. Fixed κ = 1 — rows = λ, columns = detuning
# ----------------------------------------------------------------------
plot_iv_grid(1.0)


# ----------------------------------------------------------------------
# 10. Sweep helper and its applications
# ----------------------------------------------------------------------
def iv_curve_sweep(dots, cavity, leads_template, V_array,
                   *, lam_values, kappa, dress_leads=True):
    
    """Return I(V) for each lam in lam_values, at fixed kappa.

    Builds HamiltonianBuilder + CavityDecayMatrix + TransportCalculator
    once per lam (since they don't depend on V); only TunnelRateMatrix
    is rebuilt inside the V loop because it depends on mu_L, mu_R.

    Parameters
    ----------
    dots, cavity, leads_template : the three parameter objects.
        `cavity.kappa` is overwritten with `kappa` argument.
    V_array : 1-D array of bias values to scan.
    lam_values : iterable of electron-photon couplings.
    kappa : float, cavity decay rate to use for this sweep.
    dress_leads : if True, also set leads.lam_L = leads.lam_R = lam.

    Returns
    -------
    dict {lam: np.ndarray of shape V_array.shape}
    """
    
    cavity.kappa = kappa
    results = {}
    for lam in lam_values:
        # Per-lam: Hamiltonian, eigenbasis, photon decay, transport
        ham = HamiltonianBuilder(cavity, dots, lam)
        eigvals, eigvecs, _ = ham.diagonalize()
        Gamma_ph = CavityDecayMatrix(cavity, eigvals, eigvecs).build()
        transport = TransportCalculator(cavity, leads_template)

        currents = np.empty(len(V_array))
        for k, V in enumerate(V_array):
            leads = copy.deepcopy(leads_template)
            leads.mu_L = leads.mu_0 + V
            leads.mu_R = leads.mu_0 
            if dress_leads:
                leads.lam_L = lam
                leads.lam_R = lam

            G_L, _G_R, G_tot = TunnelRateMatrix(
                cavity, dots, leads, eigvals, eigvecs
            ).build()

            P = RateEquationSolver(G_tot, Gamma_ph, cavity).steady_solver()
            currents[k] = transport.compute_current(G_L, P)

        results[lam] = currents
    return results


# ----------------------------------------------------------------------
# 10.2. λ-sweep vs κ at three detunings
# ----------------------------------------------------------------------
# 10.2: rows = kappa, columns = detuning (resonance condition); sweep lambda inside.
# lambda = 0 is the non-interacting reference (cavity decoupled => same in every row).
eps_g = 0.1                                      # bonding |g,0> kept in transport window (like 9a)
omega = 1.0
t_values = [0.4, 0.5, 0.6]                       # Delta_w = omega - 2t = +0.2, 0, -0.2
detuning_labels = [r"$\Delta\omega=+0.2$ ($t=0.4$)",
                   r"$\Delta\omega=0$ ($t=0.5$)",
                   r"$\Delta\omega=-0.2$ ($t=0.6$)"]
lam_values = [0.0, 0.5, 1.0, 1.5, 2]
kappa_rows = [0.0, 0.1, 1.0]
n_phot = 6
V_values = np.linspace(-2.0, 3.0, 600)           # fine grid -> sharp steps


def _kap(k):
    return k if k > 0 else 1e-12                 # strict 0 -> degenerate steady state


leads_template = LeadParameters(
    gamma_L=0.001, gamma_R=0.001, mu_L=0.0, mu_R=0.0, mu_0=0.0,
    T_L=0.001, T_R=0.001, lam_L=0.0, lam_R=0.0,
)

fig, axes = plt.subplots(len(kappa_rows), len(t_values),
                         figsize=(15, 4.0 * len(kappa_rows)),
                         sharex=True, sharey=True, squeeze=False)
lam_pos = [l for l in lam_values if l > 0]
colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(lam_pos)))

for r, kappa in enumerate(kappa_rows):
    for c, t in enumerate(t_values):
        ax = axes[r, c]
        e = eps_g + t                            # e1 = e2 = 0.1 + t  =>  eps_g = 0.1 for every column
        dots = DotParameters(e1=e, e2=e, t=t)
        cavity = CavityParameters(n=n_phot, omega=omega, kappa=_kap(kappa))
        curves = iv_curve_sweep(dots, cavity, leads_template, V_values,
                                lam_values=lam_values, kappa=_kap(kappa),
                                dress_leads=False)
        # non-interacting reference (lambda = 0), identical for every kappa
        ax.plot(V_values, curves[0.0], color="orange", ls="--", lw=1.1,
                label=r"$\lambda=0$ (ref)")
        for col, lam in zip(colors, lam_pos):
            ax.plot(V_values, curves[lam], color=col, lw=1.3,
                    label=rf"$\lambda={lam:g}$")
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.set_title(detuning_labels[c])
        if c == 0:
            ax.set_ylabel(rf"$\kappa={kappa:g}$" + "\n\n" + r"$I$")
        if r == len(kappa_rows) - 1:
            ax.set_xlabel(r"$V_b$")

axes[0, -1].legend(loc="lower right", fontsize=8)
fig.suptitle(r"Rabi I–V ($\varepsilon_g=0.1$ fixed) — rows = $\kappa$, columns = detuning, "
             r"$\lambda$ swept within each panel ($\lambda=0$ = non-interacting reference)", y=1.005)
fig.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "lambda_sweep_vs_kappa_grid"+".png"), dpi=150, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------
# 10.3. κ-sweep at three detunings
# ----------------------------------------------------------------------
# Same Delta_w panels, sweep kappa at fixed lambda.
eps_g = 0.1                              # bonding |g,0> kept in transport window (like 9a)
omega = 1.0
t_values = [0.4, 0.5, 0.6]
detuning_labels = [r"$\Delta\omega=+0.2$",
                   r"$\Delta\omega=0$",
                   r"$\Delta\omega=-0.2$"]
kappa_values = [1e-4, 1e-3, 1e-2, 1e-1]
lam_main = 0.5
V_values = np.linspace(-2.0, 3.0, 200)

leads_template = LeadParameters(
    gamma_L=0.002, gamma_R=0.002,
    mu_L=0.0, mu_R=0.0, mu_0=0.0,
    T_L=0.001, T_R=0.001,
    lam_L=0.0, lam_R=0.0,
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(kappa_values)))

for ax, t, dlabel in zip(axes, t_values, detuning_labels):
    e = eps_g + t                            # e1 = e2 = 0.1 + t  =>  eps_g = 0.1 for every column
    dots = DotParameters(e1=e, e2=e, t=t)
    cavity = CavityParameters(n=8, omega=omega, kappa=kappa_values[0])

    # lam=kappa=0 reference
    ref = iv_curve_sweep(
        dots, cavity, leads_template, V_values,
        lam_values=[0.0], kappa=0.0, dress_leads=False,
    )[0.0]
    ax.plot(V_values, ref, color="black", linestyle="--", lw=1.0,
            label=r"$\lambda=\kappa=0$")

    for c, kappa in zip(colors, kappa_values):
        curves = iv_curve_sweep(
            dots, cavity, leads_template, V_values,
            lam_values=[lam_main], kappa=kappa, dress_leads=False,
        )
        ax.plot(V_values, curves[lam_main], color=c, lw=1.2,
                label=rf"$\kappa={kappa:g}$")

    ax.set_xlabel(r"$V_b$")
    ax.set_title(dlabel + rf",  $t={t}$")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel(r"$I$")
axes[-1].legend(loc="lower right", fontsize=8)
fig.suptitle(rf"$\kappa$-sweep at three detunings, $\lambda={lam_main}$",
             y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "kappa_sweep_three_detunings"+".png"), dpi=150, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------
# 11. Example — single IV sweep (smoke test)
# ----------------------------------------------------------------------
# Bright-bonding smoke test: ε_g = +0.1 puts the bonding singlet above
# zero, so |0,0> can empty into the leads and the LP/UP/ε_e steps are
# clearly visible.

dots = DotParameters(e1=0.6, e2=0.6, t=0.5)
cavity = CavityParameters(n=8, omega=1.0, kappa=1e-3)
leads_template = LeadParameters(
    gamma_L=0.002, gamma_R=0.002,
    mu_L=0.0, mu_R=0.0, mu_0=0.0,
    T_L=0.001, T_R=0.001,
    lam_L=0.0, lam_R=0.0,
)

V_values = np.linspace(-2.0, 3.0, 200)

curves = iv_curve_sweep(
    dots, cavity, leads_template, V_values,
    lam_values=[0.5], kappa=1e-3, dress_leads=False,
)
I = curves[0.5]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(V_values, I, lw=1.2, label=r"$\lambda=0.5,\ \kappa=10^{-3}$")
ax.set_xlabel(r"$V_b$")
ax.set_ylabel(r"$I$")
ax.set_title(r"Bright-bonding IV — single-curve smoke test")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "smoke_test_single_iv"+".png"), dpi=150, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------
# 12. Franck–Condon lead-dressing (κ = 1)
# ----------------------------------------------------------------------
# 12. Franck-Condon lead-dressing: independent inter-dot (lam) and lead (lam_L, lam_R)
# displacements.  Built on the same one-sided-bias driver as sec 9, but with the lead
# dressings set explicitly so we can switch them on asymmetrically.

def rabi_iv_fc(dots, cavity, leads_template, lam, lam_L, lam_R, V_values):
    """One-sided-bias (mu_L = V, mu_R = 0) I(V) with independent inter-dot dressing
    `lam` and lead dressings `lam_L`, `lam_R`.  H + photon decay are built once; only
    the Fermi-dressed lead tunnel rates are rebuilt per bias point."""
    leads = copy.deepcopy(leads_template)
    leads.lam_L, leads.lam_R = lam_L, lam_R
    eigvals, eigvecs, _ = HamiltonianBuilder(cavity, dots, lam).diagonalize()
    Gamma_ph = CavityDecayMatrix(cavity, eigvals, eigvecs).build()
    transp = TransportCalculator(cavity, leads)
    I = np.empty(len(V_values))
    for j, V in enumerate(V_values):
        leads.mu_L, leads.mu_R = V, 0.0
        G_L, _G_R, G_tot = TunnelRateMatrix(cavity, dots, leads, eigvals, eigvecs).build()
        P = RateEquationSolver(G_tot, Gamma_ph, cavity).steady_solver()
        I[j] = transp.compute_current(G_L, P)
    return I


# ---- parameters: resonance (Delta_omega = 0), bright bonding (conventions of sec 9/10) ----
omega_fc = 1.0
eps_g_fc = 0.1
t_fc     = 0.5                                   # Delta_omega = omega - 2t = 0
e_fc     = eps_g_fc + t_fc                        # e1 = e2 so eps_g (bare) = 0.1
n_fc     = 10                                     # Fock cutoff N (= max photon number)
V_fc     = np.linspace(-0.5, 4.0, 800)            # 800 pts -> sharp steps

dots_fc  = DotParameters(e1=e_fc, e2=e_fc, t=t_fc)
leads_fc = LeadParameters(gamma_L=1e-3, gamma_R=1e-3, mu_L=0.0, mu_R=0.0, mu_0=0.0,
                          T_L=1e-3, T_R=1e-3, lam_L=0.0, lam_R=0.0)

# (lam, lam_L, lam_R) scenarios at kappa = 1.  (1,0,0) = interacting (inter-dot lam=1) but
# BARE leads -> isolates the effect of the LEAD dressing in the other three.
scenarios  = [(1, 0, 0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
colors_fc  = ["tab:orange", "tab:blue", "tab:green", "tab:red"]

cav_ref = CavityParameters(n=n_fc, omega=omega_fc, kappa=1e-12)   # strict 0 -> degenerate
cav_fc  = CavityParameters(n=n_fc, omega=omega_fc, kappa=1.0)

# I(V): undressed reference + the four kappa=1 scenarios
I_ref = rabi_iv_fc(dots_fc, cav_ref, leads_fc, 0.0, 0.0, 0.0, V_fc)
I_scn = {sc: rabi_iv_fc(dots_fc, cav_fc, leads_fc, *sc, V_fc) for sc in scenarios}

# ================== channel-resolved FULL rates (Gamma_L+Gamma_R)/gamma_L ==================
# Each I-V step is a charging transition switching on at its Fermi edge V = Delta_E.  The
# single-charge sector is diagonalised ONCE (the inter-dot lam=1 Hamiltonian is the same for
# every scenario; only the lead dressing differs), giving eigenstates labelled by excitation
# manifold: ground |g,0>, then polariton doublets |n-> (LPn) / |n+> (UPn).  Double-charge
# states are bare products |2,m> (doubly-occupied dot, m photons), like the empty sector |0,m>.

ev_m, evec_m, _ = HamiltonianBuilder(cav_fc, dots_fc, 1.0).diagonalize()
M_m  = n_fc + 1
s1_m = list(range(M_m, 3 * M_m)); s2_m = list(range(3 * M_m, 4 * M_m))
i00  = int(np.argmin(np.abs(ev_m[:M_m])))                       # |0,0>
ig   = min(s1_m, key=lambda f: ev_m[f])                         # |g,0>
sec1_sorted = sorted(s1_m, key=lambda f: ev_m[f])
sec2_sorted = sorted(s2_m, key=lambda f: ev_m[f])
gL = leads_fc.gamma_L

def sec1_label(k):                                              # |g,0>, |1->, |1+>, |2->, |2+>, ...
    if k == 0:
        return r"$|g,0\rangle$"
    n, sign = (k + 1) // 2, ("-" if k % 2 == 1 else "+")
    return rf"$|{n}^{{{sign}}}\rangle$"

def total_rate_rows(lam_L, lam_R, init_idx):
    """(Gamma_L+Gamma_R)[init_idx, :] / gamma_L vs V, for lead dressing (lam_L, lam_R)."""
    leads = copy.deepcopy(leads_fc); leads.lam_L, leads.lam_R = lam_L, lam_R
    R = np.empty((len(V_fc), 4 * M_m))
    for k, V in enumerate(V_fc):
        leads.mu_L, leads.mu_R = V, 0.0
        GL, GR, _ = TunnelRateMatrix(cav_fc, dots_fc, leads, ev_m, evec_m).build()
        R[k] = (GL[init_idx] + GR[init_idx]) / gL
    return R

# (1,1,0) dressed-LEFT: 1st-/2nd-electron rows ;  (1,0,0) interacting bare-leads: both rows
R00_d = total_rate_rows(1.0, 0.0, i00)
Rg0_d = total_rate_rows(1.0, 0.0, ig)
R00_i = total_rate_rows(0.0, 0.0, i00)
Rg0_i = total_rate_rows(0.0, 0.0, ig)

CUT = 5e-3                               # below this a channel is treated as identically zero

def plot_rates(ax, Rmat, idx_sorted, label_fn, ls="-", c0=0):
    """Plot total rate vs V; colour+label activating channels, gray the dark ones.
    Returns (activation list, next colour index)."""
    activ, ci = [], c0
    cmap = plt.cm.tab10
    for k, f in enumerate(idx_sorted):
        tr = Rmat[:, f]
        if tr.max() > CUT:
            ax.plot(V_fc, tr, color=cmap(ci % 10), lw=1.6, ls=ls, label=label_fn(k))
            activ.append((label_fn(k), float(tr.max()))); ci += 1
        else:
            ax.plot(V_fc, tr, color="0.85", lw=0.6, ls=ls, zorder=0)
    return activ, ci

# ================== figure: I-V, then (1,1,0) 1st/2nd-e rates, then (1,0,0) all rates ==================
fig, (axI, axA, axB, axC) = plt.subplots(4, 1, figsize=(9.5, 15), sharex=True,
                                         gridspec_kw=dict(height_ratios=[1.3, 1, 1, 1.15]))

axI.plot(V_fc, I_ref, color="black", ls="--", lw=1.5, label=r"ref: $\lambda=0,\ \kappa=0$")
for sc, col in zip(scenarios, colors_fc):
    lam, lL, lR = sc
    axI.plot(V_fc, I_scn[sc], color=col, lw=1.5,
             label=rf"$(\lambda,\lambda_L,\lambda_R)=({lam:g},{lL:g},{lR:g}),\ \kappa=1$")
axI.set_ylabel(r"$I$")
axI.set_title(r"Franck-Condon lead-dressing at $\kappa=1$ — I-V and channel activation "
              "\n"
              r"($t=0.5$, $\Delta\omega=0$, bright bonding $\varepsilon_g=0.1$, $N=10$)")
axI.grid(True, alpha=0.3); axI.legend(loc="upper left", fontsize=8)

plot_rates(axA, R00_d, sec1_sorted, sec1_label)
axA.set_ylabel(r"$(\Gamma_L+\Gamma_R)/\gamma_L$")
axA.set_title(r"DRESSED $(1,1,0)$ — 1st electron:  $|0,0\rangle \to$ single-charge eigenstates",
              fontsize=10)
axA.grid(True, alpha=0.3); axA.legend(loc="upper left", fontsize=8, ncol=2)

plot_rates(axB, Rg0_d, sec2_sorted, lambda k: rf"$|2,{k}\rangle$")
axB.set_ylabel(r"$(\Gamma_L+\Gamma_R)/\gamma_L$")
axB.set_title(r"DRESSED $(1,1,0)$ — 2nd electron:  $|g,0\rangle \to |2,m\rangle$", fontsize=10)
axB.grid(True, alpha=0.3); axB.legend(loc="upper left", fontsize=8, ncol=2)

# ---- (1,0,0): ALL rates in ONE panel.  solid = 1st electron, dashed = 2nd electron ----
a1, ci = plot_rates(axC, R00_i, sec1_sorted, sec1_label, ls="-",  c0=0)
a2, _  = plot_rates(axC, Rg0_i, sec2_sorted, lambda k: rf"$|2,{k}\rangle$", ls="--", c0=ci)
axC.set_ylabel(r"$(\Gamma_L+\Gamma_R)/\gamma_L$")
axC.set_title(r"INTERACTING, bare leads $(1,0,0)$ — ALL rates "
              r"(solid: $|0,0\rangle\to$single,  dashed: $|g,0\rangle\to|2,m\rangle$)", fontsize=10)
axC.grid(True, alpha=0.3); axC.legend(loc="upper left", fontsize=8, ncol=3)
axC.set_xlabel(r"$V_b$")

fig.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "franck_condon_lead_dressing"+".png"), dpi=150, bbox_inches="tight"); plt.show()

# ---- numeric summary: sum rule + activation heights ----
print("INTERACTING (1,0,0)  -- bare leads conserve injection weight (sum rule):")
print("   1st e- |0,0>->single :", [(l, round(h, 3)) for l, h in a1],
      " sum=%.3f" % sum(h for _, h in a1))
print("   2nd e- |g,0>->|2,m>  :", [(l, round(h, 3)) for l, h in a2])


# ----------------------------------------------------------------------
# 12 — observations: lead Franck–Condon vs inter-dot dressing
# ----------------------------------------------------------------------
# 12 — numerical check: the elastic |0,0> -> |g,0> suppression IS the Franck-Condon factor.
# Elastic injection rate / gamma = |<g,0| d_ell^dag D(lam_ell) |0,0>|^2.  If |g,0> were a pure
# product |g>(x)|0>, this equals 0.5 * e^{-lam_ell^2} (the bare FC factor, x 1/2 lead overlap).
# Verify that exactly in the product limit (inter-dot lam=0), then quantify the polaron
# correction at inter-dot lam=1 (the (1,1,0)/(1,0,1) scenarios).  Reuses sec-10 dots_fc/cav_fc.

def fc_factor_check():
    def elastic(lam_inter, lam_L, lam_R):
        """|W|^2 / gamma for the elastic |0,0> -> |g,0> step.  Reads the LEFT injection entry
        (high mu_L, f_L=1) when the left lead is the dressed/probed one, else the RIGHT
        extraction entry (mu_R=0, 1-f_R=1)."""
        ev, evec, _ = HamiltonianBuilder(cav_fc, dots_fc, lam_inter).diagonalize()
        M = n_fc + 1; s1 = range(M, 3 * M)
        i0 = int(np.argmin(np.abs(ev[:M])))          # |0,0>
        jg = min(s1, key=lambda f: ev[f])            # |g,0>
        leads = LeadParameters(gamma_L=1.0, gamma_R=1.0, mu_L=99.0, mu_R=0.0, mu_0=0.0,
                               T_L=1e-3, T_R=1e-3, lam_L=lam_L, lam_R=lam_R)
        GL, GR, _ = TunnelRateMatrix(cav_fc, dots_fc, leads, ev, evec).build()
        return GL[i0, jg] if lam_R == 0.0 else GR[jg, i0]

    print("elastic |0,0> -> |g,0>   (Gamma/gamma);   FC prediction = 0.5 * exp(-lam^2)\n")
    print("(1) CLEAN product limit: inter-dot lam = 0  ->  eigenstate is pure |g,0>")
    print(f"   {'lam_L':>6} {'Gamma/gamma':>12} {'0.5 e^-lamL^2':>14} {'ratio':>8}")
    for lL in [0.0, 0.5, 1.0, 1.5, 2.0]:
        r = elastic(0.0, lL, 0.0); fc = 0.5 * np.exp(-lL**2)
        print(f"   {lL:>6} {r:>12.5f} {fc:>14.5f} {r/fc:>8.4f}")

    print("\n(2) POLARON regime: inter-dot lam = 1   (the (1,1,0)/(1,0,1) scenarios)")
    b  = elastic(1.0, 0.0, 0.0)                      # undressed  (= (1,0,0))
    dL = elastic(1.0, 1.0, 0.0)                      # left  dressed (1,1,0)
    dR = elastic(1.0, 0.0, 1.0)                      # right dressed (1,0,1)
    print(f"   undressed  |W|^2              = {b:.4f}")
    print(f"   left  dressed (lam_L=1)       = {dL:.4f}    ratio dL/b = {dL/b:.4f}")
    print(f"   right dressed (lam_R=1)       = {dR:.4f}    ratio dR/b = {dR/b:.4f}")
    print(f"   pure FC factor  e^-lam^2      = {np.exp(-1.0):.4f}")
    print(f"   geometric mean  sqrt(dL dR)/b = {np.sqrt(dL * dR) / b:.4f}"
          "   (+/- polaron interference cancels -> recovers e^-1)")

fc_factor_check()

"""
jc_wc.py — Jaynes–Cummings (weak-coupling) two-dot transport stack.

Extracted verbatim from wk_coupl_refactored.ipynb (cells 1, 2, 4-9) so it can
be imported from any notebook in correcting_scripts/ without re-executing the
notebook. Convention reminder: JCBuilder's `lam` argument IS the photon
coupling g; to compare against the polaron-Rabi model at polaron coupling
lambda, call JCBuilder(cavity, dots, t * lambda).

`iv_curve(...)` sweeps the source-lead chemical potential one-sided
(mu_source = V, other lead = 0).
"""



import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


# ============================================================
# Parameter containers
# ============================================================

class DotParameters:
    """Two-dot system: site energies (e1, e2) and inter-dot tunnel coupling t."""

    def __init__(self, e1, e2, t):
        self.e1 = e1
        self.e2 = e2
        self.t = t


class CavityParameters:
    """Cavity: photon cutoff n (Fock dim = n + 1), frequency omega, photon loss kappa."""

    def __init__(self, n, omega, kappa):
        self.n = n
        self.omega = omega
        self.kappa = kappa


class LeadParameters:
    """Lead reservoirs: bare couplings, chemical potentials, temperatures, and
    e-phonon (cavity) displacements lambda_L/R that dress the tunneling operators."""

    def __init__(self, gamma_L, gamma_R, mu_L, mu_R, mu_0,
                 T_L, T_R, lambda_L, lambda_R):
        self.gamma_L = gamma_L
        self.gamma_R = gamma_R
        self.mu_L = mu_L
        self.mu_R = mu_R
        self.mu_0 = mu_0
        self.T_L = T_L
        self.T_R = T_R
        self.lambda_L = lambda_L
        self.lambda_R = lambda_R

    def fermi(self, E, lead="L"):
        """Fermi–Dirac occupation in the requested lead. Handles T = 0 explicitly
        and clips the exponent to avoid overflow at finite (but small) T."""
        E = np.asarray(E, dtype=float)
        mu = self.mu_L if lead == "L" else self.mu_R
        T = self.T_L if lead == "L" else self.T_R
        if T == 0:
            return np.where(E < mu, 1.0, 0.0)
        x = np.clip((E - mu) / T, -700.0, 700.0)
        return 1.0 / (np.exp(x) + 1.0)


# ============================================================
# Hamiltonian builder
# ============================================================

class JCBuilder:
    """Block-diagonal Hamiltonian over four charge sectors.

    Basis ordering (M = cavity.n + 1):
        sector 0  (M states):   |0, 0>, |0, 1>, ..., |0, M-1>
        sector 1  (2M states):  |g, 0>, |e, 0>, |g, 1>, |e, 1>, ...   (interleaved)
        sector 2  (M states):   |2, 0>, |2, 1>, ..., |2, M-1>
    """

    def __init__(self, cavity: CavityParameters, dots: DotParameters, lam):
        self.cavity = cavity
        self.dots = dots
        self.lam = lam

        self.M = cavity.n + 1
        self.dim = 4 * self.M

        self.H0 = self._sector0()
        self.H1 = self._sector1()
        self.H2 = self._sector2()
        self.H = self._assemble()

    # -------------------------------------------------
    # Sector blocks
    # -------------------------------------------------

    def _sector0(self):
        return np.diag(self.cavity.omega * np.arange(self.M)).astype(float)

    def _sector2(self):
        e_sum = self.dots.e1 + self.dots.e2
        return np.diag(e_sum + self.cavity.omega * np.arange(self.M)).astype(float)

    def _sector1(self):
        avg = (self.dots.e1 + self.dots.e2) / 2
        split = np.sqrt((self.dots.e1 - self.dots.e2)**2 / 4 + self.dots.t**2)
        eg = avg - split
        ee = avg + split

        H1 = np.zeros((2 * self.M, 2 * self.M), dtype=float)
        for n in range(self.M):
            H1[2*n,     2*n    ] = eg + self.cavity.omega * n     # |g, n>
            H1[2*n + 1, 2*n + 1] = ee + self.cavity.omega * n     # |e, n>

        # JC coupling: λ (σ_+ a + σ_- a†) connects |e, m> ↔ |g, m+1>
        for m in range(self.cavity.n):
            r = 2*m + 1          # |e, m>
            c = r + 1            # |g, m+1>
            H1[r, c] = np.sqrt(m + 1) * self.lam
            H1[c, r] = np.sqrt(m + 1) * self.lam
        return H1

    def _assemble(self):
        M, dim = self.M, self.dim
        H = np.zeros((dim, dim), dtype=float)
        H[:M, :M] = self.H0
        H[M:3*M, M:3*M] = self.H1
        H[3*M:, 3*M:] = self.H2
        return H

    # -------------------------------------------------
    # Diagonalisation
    # -------------------------------------------------

    def diagonalize(self, ordered=True):
        """Block-diagonalise sector-by-sector and (optionally) sort each JC eigenstate
        by its dominant overlap with the bare basis, so the column index keeps a
        physical labelling across parameter sweeps.
        """
        M, dim = self.M, self.dim

        eigvals_0 = np.diag(self.H0)
        eigvals_2 = np.diag(self.H2)
        eigvecs_0 = np.eye(M)
        eigvecs_2 = np.eye(M)

        eigvals_1, eigvecs_1 = eigh(self.H1)

        if ordered:
            dominant = np.argmax(np.abs(eigvecs_1)**2, axis=0)
            order = np.argsort(dominant)
            eigvals_1 = eigvals_1[order]
            eigvecs_1 = eigvecs_1[:, order]

        V0 = np.zeros((dim, M),    dtype=complex)
        V1 = np.zeros((dim, 2*M),  dtype=complex)
        V2 = np.zeros((dim, M),    dtype=complex)
        V0[:M,       :] = eigvecs_0
        V1[M:3*M,    :] = eigvecs_1
        V2[3*M:,     :] = eigvecs_2

        eigvals = np.concatenate([eigvals_0, eigvals_1, eigvals_2])
        eigvecs = np.hstack([V0, V1, V2])
        charges = [0]*M + [1]*(2*M) + [2]*M
        return eigvals, eigvecs, charges


# ============================================================
# Tunneling operators and Fermi-golden-rule rates
# ============================================================

def _hamiltonian_basis_permutation(M):
    """Permutation P : (charge ⊗ phonon) kron basis  →  Hamiltonian basis.

    The kron basis is ordered with charge ∈ {|0>, |g>, |e>, |2>} outer and
    phonon n inner; the Hamiltonian uses interleaved |g, n>, |e, n> inside
    sector 1. P[new, old] = 1 maps column `old` to row `new`.
    """
    dim = 4 * M
    P = np.zeros((dim, dim))
    for n in range(M):
        P[n, n] = 1.0                                  # |0, n>
        P[M + 2*n,     M + n        ] = 1.0            # |g, n>
        P[M + 2*n + 1, M + M + n    ] = 1.0            # |e, n>
        P[3*M + n,     3*M + n      ] = 1.0            # |2, n>
    return P


class JCDisplacementOperator:
    """Linearised lead displacement on the Fock space: D = I - λ (a - a†)."""

    @staticmethod
    def matrix(N, lam):
        dim = N + 1
        a = np.zeros((dim, dim))
        for n in range(1, dim):
            a[n - 1, n] = np.sqrt(n)
        adag = a.T
        return np.eye(dim) - lam * (a - adag)


class JCElectronOperators:
    """Lead annihilation operators d_L, d_R in the dressed dot basis {|0>, |g>, |e>, |2>}."""

    def __init__(self, dots: DotParameters):
        self.dots = dots

    def angles(self):
        alpha = np.arctan2(-2 * self.dots.t, self.dots.e1 - self.dots.e2) / 2
        return alpha, np.cos(alpha), np.sin(alpha)

    def annihilation(self):
        _, c, s = self.angles()

        d_g = np.zeros((4, 4))
        d_g[0, 1] = 1.0
        d_g[2, 3] = 1.0

        d_e = np.zeros((4, 4))
        d_e[0, 2] = 1.0
        d_e[1, 3] = -1.0

        d_l =  c * d_e + s * d_g
        d_r = -s * d_e + c * d_g
        return d_l, d_r


class JCTunnelOperators:
    """V_α = d_α ⊗ D_α, permuted from the kron basis to the Hamiltonian basis."""

    def __init__(self, cavity: CavityParameters, dots: DotParameters,
                 leads: LeadParameters):
        self.cavity = cavity
        self.dots = dots
        self.leads = leads
        self.M = cavity.n + 1

    def build(self):
        d_l, d_r = JCElectronOperators(self.dots).annihilation()
        D_L = JCDisplacementOperator.matrix(self.cavity.n, self.leads.lambda_L)
        D_R = JCDisplacementOperator.matrix(self.cavity.n, self.leads.lambda_R)

        V_l_ann = np.kron(d_l, D_L)
        V_r_ann = np.kron(d_r, D_R)

        P = _hamiltonian_basis_permutation(self.M)
        V_l_ann = P @ V_l_ann @ P.T
        V_r_ann = P @ V_r_ann @ P.T

        return V_l_ann, V_r_ann, V_l_ann.conj().T, V_r_ann.conj().T


class JCTunnelRate:
    """Lead-resolved Fermi-golden-rule rate matrices in the system eigenbasis.

    Convention: ``Γ[f, i] = rate(i → f)``.

    The lead operators (in the eigenbasis) are cached at construction; only the
    Fermi functions are re-evaluated on each ``build()`` call. To sweep a bias,
    update ``self.leads.mu_L`` / ``self.leads.mu_R`` and call ``build()`` again.
    """

    def __init__(self, cavity: CavityParameters, dots: DotParameters,
                 leads: LeadParameters, eigvals, eigvecs):
        self.cavity = cavity
        self.dots = dots
        self.leads = leads
        self.eigvals = eigvals
        self.eigvecs = eigvecs

        self.M = cavity.n + 1
        self.dim = 4 * self.M
        self.idx0 = range(0, self.M)
        self.idx1 = range(self.M, 3 * self.M)
        self.idx2 = range(3 * self.M, 4 * self.M)

        self._cache_operators_in_eigenbasis()

    def _cache_operators_in_eigenbasis(self):
        V_l_ann, V_r_ann, V_l_cre, V_r_cre = JCTunnelOperators(
            self.cavity, self.dots, self.leads
        ).build()
        Udag = self.eigvecs.conj().T
        U = self.eigvecs
        self.V_l_ann = Udag @ V_l_ann @ U
        self.V_r_ann = Udag @ V_r_ann @ U
        self.V_l_cre = Udag @ V_l_cre @ U
        self.V_r_cre = Udag @ V_r_cre @ U

    def _rate(self, i, f):
        Ei = self.eigvals[i]
        Ef = self.eigvals[f]
        dE = Ef - Ei

        i0, i1, i2 = self.idx0, self.idx1, self.idx2
        if (i in i0 and f in i1) or (i in i1 and f in i2):
            adding = True
        elif (i in i1 and f in i0) or (i in i2 and f in i1):
            adding = False
        else:
            return 0.0, 0.0

        if adding:
            WL, WR = self.V_l_cre[f, i], self.V_r_cre[f, i]
            fL = self.leads.fermi(dE, "L")
            fR = self.leads.fermi(dE, "R")
        else:
            WL, WR = self.V_l_ann[f, i], self.V_r_ann[f, i]
            fL = 1.0 - self.leads.fermi(-dE, "L")
            fR = 1.0 - self.leads.fermi(-dE, "R")

        rL = self.leads.gamma_L * abs(WL)**2 * fL
        rR = self.leads.gamma_R * abs(WR)**2 * fR
        return rL, rR

    def build(self):
        Γ_L = np.zeros((self.dim, self.dim), dtype=float)
        Γ_R = np.zeros((self.dim, self.dim), dtype=float)

        for sec_lo, sec_hi in [(self.idx0, self.idx1), (self.idx1, self.idx2)]:
            for i in sec_lo:
                for f in sec_hi:
                    rL, rR = self._rate(i, f)
                    Γ_L[f, i], Γ_R[f, i] = rL, rR
                    rL, rR = self._rate(f, i)
                    Γ_L[i, f], Γ_R[i, f] = rL, rR

        return Γ_L, Γ_R, Γ_L + Γ_R


# ============================================================
# Cavity decay (photon loss)
# ============================================================

class JCCavityDecay:
    """Photonic dissipator: Γ_ph[f, i] = κ |⟨f|a|i⟩|²  for  E_f < E_i."""

    def __init__(self, cavity: CavityParameters, eigvals, eigvecs):
        self.cavity = cavity
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.M = cavity.n + 1
        self.dim = 4 * self.M
        self.kappa = cavity.kappa
        self._cache_photon_operator()

    def _photon_lab(self):
        a_ph = np.zeros((self.M, self.M))
        for m in range(1, self.M):
            a_ph[m - 1, m] = np.sqrt(m)
        I_charge = np.eye(4)
        a_kron = np.kron(I_charge, a_ph)
        P = _hamiltonian_basis_permutation(self.M)
        return P @ a_kron @ P.T

    def _cache_photon_operator(self):
        a = self._photon_lab()
        Udag = self.eigvecs.conj().T
        self.a = Udag @ a @ self.eigvecs

    def build(self):
        Ef = self.eigvals[:, None]          # row index = f
        Ei = self.eigvals[None, :]          # col index = i
        mask = (Ef < Ei).astype(float)
        amp2 = np.abs(self.a)**2
        return self.kappa * amp2 * mask


# ============================================================
# Classical rate equation
# ============================================================

class JCRateEquation:
    """Master equation dP/dt = L P built from the total transition matrix Γ.

    Off-diagonal:  L[f, i] = Γ[f, i] = rate(i → f).
    Diagonal:      L[i, i] = -Σ_f Γ[f, i]  (total outgoing rate from i).
    """

    def __init__(self, Γ_tot, Γ_ph, cavity: CavityParameters):
        self.cavity = cavity
        self.Γ_full = Γ_tot + Γ_ph

        self.M = cavity.n + 1
        self.n0 = self.M
        self.n1 = 2 * self.M
        self.n2 = self.M
        self.dim = self.n0 + self.n1 + self.n2

        self.P0 = slice(0, self.n0)
        self.P1 = slice(self.n0, self.n0 + self.n1)
        self.P2 = slice(self.n0 + self.n1, self.dim)

        self.L = self._build_master()

    def _build_master(self):
        L = np.asarray(self.Γ_full, dtype=float).copy()
        outgoing = self.Γ_full.sum(axis=0)
        np.fill_diagonal(L, -outgoing)
        return L

    def blocks(self):
        L = self.L
        return {
            "A": L[self.P0, self.P0],
            "B": L[self.P0, self.P1],
            "D": L[self.P1, self.P0],
            "E": L[self.P1, self.P1],
            "F": L[self.P1, self.P2],
            "H": L[self.P2, self.P1],
            "I": L[self.P2, self.P2],
        }

    def steady_solver(self):
        """Steady state via the zero eigenvalue of L, normalised to ΣP = 1."""
        evals, evecs = np.linalg.eig(self.L)
        idx = np.argmin(np.abs(evals))
        P = np.real(evecs[:, idx])
        total = P.sum()
        if total < 0:                        # flip if eigenvector came back negative
            P, total = -P, -total
        return P / total


# ============================================================
# Transport observables
# ============================================================

class JCTransport:
    """Currents and differential conductances from steady-state populations."""

    def __init__(self, cavity: CavityParameters, leads: LeadParameters):
        self.cavity = cavity
        self.leads = leads

        M = cavity.n + 1
        self.n0 = M
        self.n1 = 2 * M
        self.n2 = M
        self.dim = self.n0 + self.n1 + self.n2

        self.i0 = slice(0, self.n0)
        self.i1 = slice(self.n0, self.n0 + self.n1)
        self.i2 = slice(self.n0 + self.n1, self.dim)

    def compute_current(self, Γ_alpha, P):
        """Lead-α current from the rate matrix Γ_α (with Γ[f, i] = rate(i → f))
        and the steady-state vector P. Normalisation factor γ_L γ_R / (γ_L + γ_R)
        matches the convention used in the IV plots.
        """
        if Γ_alpha.shape != (self.dim, self.dim):
            raise ValueError("Gamma matrix has wrong dimension.")
        if P.shape[0] != self.dim:
            raise ValueError("Probability vector has wrong dimension.")

        P0, P1, P2 = P[self.i0], P[self.i1], P[self.i2]

        G01 = Γ_alpha[self.i1, self.i0]   # 0 → 1  (adding from sector 0)
        G10 = Γ_alpha[self.i0, self.i1]   # 1 → 0  (removing toward sector 0)
        G12 = Γ_alpha[self.i2, self.i1]   # 1 → 2  (adding from sector 1)
        G21 = Γ_alpha[self.i1, self.i2]   # 2 → 1  (removing toward sector 1)

        term_add01 = (P0 @ G01.sum(axis=0)).real
        term_net1  = (P1 @ (G12 - G10).sum(axis=0)).real
        term_rem21 = (P2 @ G21.sum(axis=0)).real

        gL, gR = self.leads.gamma_L, self.leads.gamma_R
        norm = gL * gR / (gL + gR)
        return (term_add01 + term_net1 - term_rem21) / norm

    def differential_conductance(self, V_values, current_function):
        """G(V) = (γ_L + γ_R) · dI/dV, evaluated by finite differences."""
        I = np.asarray(current_function(V_values), dtype=float)
        gL, gR = self.leads.gamma_L, self.leads.gamma_R
        return (gL + gR) * np.gradient(I, V_values)


# ============================================================
# Bias-sweep helper
# ============================================================

def iv_curve(dots: DotParameters,
             cavity: CavityParameters,
             leads_template: LeadParameters,
             lam: float,
             V_values,
             source_lead: str = "L"):
    """I(V) at fixed (dots, cavity, lam), sweeping the chemical potential of
    ``source_lead`` over ``V_values`` (the other lead is held at 0).

    The Hamiltonian, cavity decay matrix, and tunneling operators in the
    eigenbasis are built once and reused for every V; only the Fermi-dressed
    rate matrices are rebuilt inside the loop.
    """
    leads = copy.deepcopy(leads_template)

    builder = JCBuilder(cavity, dots, lam)
    eigvals, eigvecs, _ = builder.diagonalize(ordered=True)

    Γ_ph = JCCavityDecay(cavity, eigvals, eigvecs).build()
    rates = JCTunnelRate(cavity, dots, leads, eigvals, eigvecs)
    transp = JCTransport(cavity, leads)

    currents = np.empty(len(V_values), dtype=float)
    for j, V in enumerate(V_values):
        if source_lead == "L":
            leads.mu_L, leads.mu_R = V, 0.0
        else:
            leads.mu_L, leads.mu_R = 0.0, V

        Γ_L, Γ_R, Γ_tot = rates.build()
        P = JCRateEquation(Γ_tot, Γ_ph, cavity).steady_solver()

        Γ_alpha = Γ_L if source_lead == "L" else Γ_R
        currents[j] = transp.compute_current(Γ_alpha, P)

    return currents

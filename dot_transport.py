"""
Two-quantum-dot Rabi transport model (strong coupling).

Core library: parameter containers, operators, Hamiltonian builder, Lindblad/rate
master equation, and transport observables for a double quantum dot coupled to a
single cavity mode in the polaron (displaced-Fock) basis.

Importable model classes used by analysis.py.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.special import eval_genlaguerre, factorial


# ======================================================================
# 1a. `DotParameters` — dot energies and inter-dot hopping
# ======================================================================
class DotParameters:
    def __init__(self, e1, e2, t):
        
        self.e1 = e1
        self.e2 = e2
        self.t = t


# ======================================================================
# 1b. `CavityParameters` — photon cutoff, frequency, decay rate κ
# ======================================================================
class CavityParameters:
    def __init__(self, n, omega, kappa):
        
        self.n = n
        self.omega = omega
        self.kappa = kappa


# ======================================================================
# 1c. `LeadParameters` — chemical potentials, temperatures, lead-dressing λ_L/R
# ======================================================================
class LeadParameters:
    def __init__(self, gamma_L, gamma_R, mu_L, mu_R, mu_0, T_L, T_R, lam_L, lam_R):
        
        self.gamma_L = gamma_L
        self.gamma_R = gamma_R
        self.mu_L = mu_L
        self.mu_R = mu_R
        self.mu_0 = mu_0
        self.T_L = T_L
        self.T_R = T_R
        self.lam_L = lam_L
        self.lam_R = lam_R
        
    # -------------------------------------------------
    # Fermi function belongs naturally to the lead
    # -------------------------------------------------

    def fermi(self, E, lead="L"):
        """
        Fermi distribution for left or right lead.
        """
        E = np.array(E)
        if lead == "L":
            mu = self.mu_L
            T = self.T_L
        else:
            mu = self.mu_R
            T = self.T_R
        # Zero temperature
        if T == 0:
            return np.where(E < mu, 1.0, 0.0)
        # Finite temperature
        x = (E - mu) / T
        x = np.clip(x, -700, 700)  # prevent overflow
        return 1.0 / (np.exp(x) + 1.0)


# ======================================================================
# 2a. `PhotonicOperators` — a, a†, n̂ in cavity Fock basis
# ======================================================================
class PhotonicOperators:
    def __init__(self, cavity: CavityParameters):
        """
        Constructs photonic operators a and a^\dagger
        in the combined charge × photon Hilbert space.

        Parameters
        ----------
        n_photons : int
            Maximum photon number (truncated at n_photons),
            giving photon space dimension n_photons + 1.
        charge_dim : int
            Total dimension of the charge space (default 4).
        """
        self.cavity = cavity
        self.n = self.cavity.n
        self.photon_dim = self.n + 1

        # Identity in charge space
        self.I_charge = np.eye(4)

        # Photon operators in photon space
        self.a_photon = self._create_a_photon()
        self.adag_photon = self._create_adag_photon()

        # Lift to total charge × photon space
        self.a = np.kron(self.I_charge, self.a_photon)
        self.adag = np.kron(self.I_charge, self.adag_photon)

    def _create_a_photon(self):
        """Photon annihilation operator in truncated Fock basis."""
        a = np.zeros((self.photon_dim, self.photon_dim))
        for m in range(1, self.photon_dim):
            a[m-1, m] = np.sqrt(m)
        return a

    def _create_adag_photon(self):
        """Photon creation operator in truncated Fock basis."""
        adag = np.zeros((self.photon_dim, self.photon_dim))
        for m in range(self.photon_dim - 1):
            adag[m+1, m] = np.sqrt(m+1)
        return adag


# ======================================================================
# 2b. `ElectronOperators` — d_L, d_R in the 4-state dot basis
# ======================================================================
class ElectronOperators:
    """
    4×4 charge operators per phonon number:
    basis = |0>, |L>, |R>, |2>
    """
    
    @staticmethod
    def annihilation():
        d_l = np.zeros((4, 4), dtype=np.complex128)
        d_r = np.zeros((4, 4), dtype=np.complex128)
    
        d_l[0, 1] = 1.0     # <0| d_L |L>
        d_l[2, 3] = 1.0     # <R| d_L |2>
    
        d_r[0, 2] = 1.0     # <0| d_R |R>
        d_r[1, 3] = -1.0    # <L| d_R |2>
    
        return d_l, d_r


# ======================================================================
# 2c. `TunnelOperators` — dressed tunneling V_ℓ = D̂(λ_ℓ) ⊗ d_ℓ
# ======================================================================
class TunnelOperators:
    """
    Construct full tunneling operators for left and right leads
    including displacement (electron-photon) operators.
    """
    def __init__(self, cavity: CavityParameters, leads: LeadParameters):
        """
        Args:
            cavity: CavityParameters object (defines phonon cutoff)
            lead_params: LeadParameters object (contains lam_L, lam_R)
        """
        self.cavity = cavity
        self.leads = leads
        self.M = self.cavity.n + 1  # phonon dimension
        self.lam_L = leads.lam_L
        self.lam_R = leads.lam_R

    def build(self):
        """
        Returns:
            V_l_ann, V_r_ann, V_l_cre, V_r_cre 
            Full tunneling operators including phonons.
        """
        d_l, d_r = ElectronOperators.annihilation()

        # Displacement operators (phonon part)
        D_L = DisplacementOperator.matrix(self.cavity.n, self.leads.lam_L)
        D_R = DisplacementOperator.matrix(self.cavity.n, self.leads.lam_R)

        # Full operators in combined Hilbert space
        V_l_ann = np.kron(d_l, D_L)
        V_r_ann = np.kron(d_r, D_R)

        # Creation operators are Hermitian transpose
        V_l_cre = V_l_ann.T.conj()
        V_r_cre = V_r_ann.T.conj()

        return V_l_ann, V_r_ann, V_l_cre, V_r_cre


# ======================================================================
# 2d. `DisplacementOperator` — D̂(λ) = exp[λ(a − a†)] matrix elements
# ======================================================================
class DisplacementOperator:
    @staticmethod
    def element(m, n, lam):
        s = np.exp(-lam**2 / 2)
        if m >= n:
            k = m - n
            return (
                s
                * np.sqrt(factorial(n) / factorial(m))
                * lam**k
                * eval_genlaguerre(n, k, lam**2)
            )
        else:
            k = n - m
            return (
                s
                * np.sqrt(factorial(m) / factorial(n))
                * (-lam)**k
                * eval_genlaguerre(m, k, lam**2)
            )
    @staticmethod
    def matrix(N, lam):
        return np.array(
            [[DisplacementOperator.element(m, n, lam) for n in range(N + 1)]
             for m in range(N + 1)],
            dtype=complex
        )


# ======================================================================
# 3. Hamiltonian
# ======================================================================
class HamiltonianBuilder:
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
    
    """Now let's build the Hamiltonian step-by-step."""
    # -----------------------------
    # Sector builders
    # -----------------------------

    def _build_sector_0(self):
        """Constructs the zero-particle sector Hamiltonian."""
        H0 = np.zeros((self.M, self.M), dtype=np.complex128)
        for n in range(self.M):
            H0[n, n] = self.cavity.omega * n
        return H0

    def _build_sector_2(self):
        """Constructs the two-particle sector Hamiltonian."""
        H2 = np.zeros((self.M, self.M), dtype=np.complex128)
        for n in range(self.M):
            H2[n, n] = self.dots.e1 + self.dots.e2 + self.cavity.omega * n
        return H2

    def _build_sector_1(self):
        """Constructs the single-particle sector Hamiltonian."""
        H1 = np.zeros((2 * self.M, 2 * self.M), dtype=np.complex128)

        # L block
        for n in range(self.M):
            H1[n, n] = self.dots.e1 + self.cavity.omega * n

        # R block
        for n in range(self.M):
            H1[self.M + n, self.M + n] = self.dots.e2 + self.cavity.omega * n

        # Hopping
        for n in range(self.M):
            for m in range(self.M):
                iL = n
                iR = self.M + m

                H1[iL, iR] = (
                    self.dots.t
                    * DisplacementOperator.element(m, n, self.lam)
                )
                H1[iR, iL] = (
                    self.dots.t
                    * DisplacementOperator.element(n, m, -self.lam)
                )

        return (H1 + H1.conj().T) / 2.0

    # -----------------------------
    # Full Hamiltonian
    # -----------------------------

    def build(self):
        H0 = self._build_sector_0()
        H1 = self._build_sector_1()
        H2 = self._build_sector_2()

        H = np.zeros((self.dim, self.dim), dtype=np.complex128)

        idx0 = list(range(self.M))
        idx1 = list(range(self.M, self.M + 2 * self.M))
        idx2 = list(range(self.M + 2 * self.M, self.M + 3 * self.M))

        # Fill blocks
        H[np.ix_(idx0, idx0)] = H0
        H[np.ix_(idx1, idx1)] = H1
        H[np.ix_(idx2, idx2)] = H2

        return (H + H.conj().T) / 2.0

    # -----------------------------
    # Diagonalization
    # -----------------------------

    def diagonalize(self):
        """
        Diagonalize Hamiltonian and return eigenvalues/eigenvectors
        sorted by charge sector (0 → 1 → 2), and energy-sorted inside each sector).
        """
    
        eigvals, eigvecs = eigh(self.H)
    
        # --- Build charge operator ---
        Q = np.zeros((self.dim, self.dim))
        M = self.M
    
        Q[0:M, 0:M] = 0
        Q[M:3*M, M:3*M] = np.eye(2*M) * 1
        Q[3*M:4*M, 3*M:4*M] = np.eye(M) * 2
    
        # --- Compute charge expectation values ---
        charge_vals = [
            np.real(v.conj().T @ (Q @ v))
            for v in eigvecs.T
        ]
    
        # --- Collect indices by sector ---
        sector_indices = {0: [], 1: [], 2: []}
        tol = 1e-8
    
        for i, q in enumerate(charge_vals):
            for sector in [0, 1, 2]:
                if abs(q - sector) < tol:
                    sector_indices[sector].append(i)
                    break
    
        # --- Sort each sector internally by energy ---
        for sector in sector_indices:
            sector_indices[sector].sort(key=lambda i: eigvals[i])
    
        ordered_indices = (
            sector_indices[0]
            + sector_indices[1]
            + sector_indices[2]
        )
    
        eigvals_sorted = eigvals[ordered_indices]
        eigvecs_sorted = eigvecs[:, ordered_indices]
    
        state_charge = (
            [0]*len(sector_indices[0]) +
            [1]*len(sector_indices[1]) +
            [2]*len(sector_indices[2])
        )
    
        return eigvals_sorted, eigvecs_sorted, state_charge


# ======================================================================
# 4a. `CavityDecayMatrix` — Lindblad photon-decay rate matrix Γ_ph
# ======================================================================
class CavityDecayMatrix:

    def __init__(self, cavity, eigvals, eigvecs):
        self.cavity = cavity
        self.eigvals = eigvals
        self.eigvecs = eigvecs

        self.M = cavity.n + 1
        self.dim = 4 * self.M
        self.kappa = cavity.kappa

        self._transform_ph_operators()

    def _transform_ph_operators(self):

        ph_op = PhotonicOperators(self.cavity)

        Udag = self.eigvecs.conj().T
        U = self.eigvecs

        self.a = Udag @ ph_op.a @ U
        self.adag = Udag @ ph_op.adag @ U

    def _rate(self, i, f):

        Ei = self.eigvals[i]
        Ef = self.eigvals[f]

        # photon decay: must lower energy
        if Ef >= Ei:
            return 0.0

        W = self.a[f, i]
        return (abs(W)**2)*self.kappa

    def build(self):

        # Store as Γ_ph[i, f] = rate(i → f), the SAME [initial, final] convention
        # as TunnelRateMatrix, so RateEquationSolver (which does L = Γ_full.T) treats
        # photon decay and lead tunneling consistently. (Storing [f, i] here would
        # transpose only the decay term, turning emission into absorption and pumping
        # the cavity up — a bug that is invisible at κ→0 but dominates at finite κ.)
        Γ_ph = np.zeros((self.dim, self.dim))

        for i in range(self.dim):
            for f in range(self.dim):
                Γ_ph[i, f] = self._rate(i, f)

        return Γ_ph


# ======================================================================
# 4b. `TunnelRateMatrix` — Fermi-dressed lead tunneling rates Γ_L, Γ_R
# ======================================================================
class TunnelRateMatrix:
    """
    Builds Γ_L, Γ_R, Γ_total in the eigenbasis,
    including full tunneling matrix elements.
    """

    def __init__(
        self,
        cavity: CavityParameters,
        dots: DotParameters,
        leads: LeadParameters,
        eigvals,
        eigvecs
        ):

        self.cavity = cavity
        self.dots = dots
        self.leads = leads

        self.eigvals = eigvals
        self.eigvecs = eigvecs

        # Hilbert space sizes
        self.M = self.cavity.n + 1
        self.dim = 4 * self.M

        # Sector index ranges (guaranteed ordered)
        self.idx0 = range(0, self.M)
        self.idx1 = range(self.M, 3 * self.M)
        self.idx2 = range(3 * self.M, 4 * self.M)

        # -----------------------------
        # Tunnel operators
        # -----------------------------
        
        self._transform_tunneling_operators()

    # -------------------------------------------------
    # Build and rotate tunneling operators
    # -------------------------------------------------
    def _transform_tunneling_operators(self):

        tunnel = TunnelOperators(self.cavity, self.leads)
        V_l_ann, V_r_ann, V_l_cre, V_r_cre = tunnel.build()

        # Transform to eigenbasis
        Udag = self.eigvecs.conj().T
        U = self.eigvecs

        self.V_l_ann = Udag @ V_l_ann @ U
        self.V_r_ann = Udag @ V_r_ann @ U
        self.V_l_cre = Udag @ V_l_cre @ U
        self.V_r_cre = Udag @ V_r_cre @ U

    # -------------------------------------------------
    # Single transition rate
    # -------------------------------------------------
    def _rate(self, i, f):
        """
        Compute Γ_L and Γ_R for transition i → f.
        """

        Ei = self.eigvals[i]
        Ef = self.eigvals[f]
        ΔE = Ef - Ei

        # Determine if particle is added or removed
        # (sector ordering guarantees this works)
        if i in self.idx0 and f in self.idx1:
            adding = True
        elif i in self.idx1 and f in self.idx2:
            adding = True
        elif i in self.idx1 and f in self.idx0:
            adding = False
        elif i in self.idx2 and f in self.idx1:
            adding = False
        else:
            return 0.0, 0.0  # forbidden transition

        if adding:
            WL = self.V_l_cre[f, i]
            WR = self.V_r_cre[f, i]

            fL = self.leads.fermi(ΔE, lead="L")
            fR = self.leads.fermi(ΔE, lead="R")

        else:
            WL = self.V_l_ann[f, i]
            WR = self.V_r_ann[f, i]

            # Note: ΔE = Ef - Ei < 0 for removal
            fL = 1 - self.leads.fermi(-ΔE, lead="L")
            fR = 1 - self.leads.fermi(-ΔE, lead="R")

        γL = self.leads.gamma_L
        γR = self.leads.gamma_R

        rL = γL * abs(WL) ** 2 * fL if WL != 0 else 0.0
        rR = γR * abs(WR) ** 2 * fR if WR != 0 else 0.0

        return rL, rR

    # -------------------------------------------------
    # Build full Γ matrices
    # -------------------------------------------------
    def build(self):

        Γ_L = np.zeros((self.dim, self.dim))
        Γ_R = np.zeros((self.dim, self.dim))

        # -------- 0 ↔ 1 transitions ----------
        for i in self.idx0:
            for f in self.idx1:
                rL, rR = self._rate(i, f)
                Γ_L[i, f] = rL
                Γ_R[i, f] = rR

                rL, rR = self._rate(f, i)
                Γ_L[f, i] = rL
                Γ_R[f, i] = rR

        # -------- 1 ↔ 2 transitions ----------
        for i in self.idx1:
            for f in self.idx2:
                rL, rR = self._rate(i, f)
                Γ_L[i, f] = rL
                Γ_R[i, f] = rR

                rL, rR = self._rate(f, i)
                Γ_L[f, i] = rL
                Γ_R[f, i] = rR

        Γ_total = Γ_L + Γ_R

        return Γ_L, Γ_R, Γ_total


# ======================================================================
# 4c. `RateEquationSolver` — steady-state population of eigenstates
# ======================================================================
class RateEquationSolver:
    """
    Builds and solves the classical master equation

        dP/dt = L P

    from the total transition matrix Γ_tot.

    Keeps exact sector block structure:

           [ A   B   0 ]
    L  =   [ D   E   F ]
           [ 0   H   I ]
    """

    def __init__(self, Γ_tot, Γ_ph, cavity: CavityParameters):
        self.Γ_tot = Γ_tot
        self.Γ_ph = Γ_ph
        self.cavity = cavity
        self.Γ_full = self.Γ_tot + self.Γ_ph

        self.m = self.cavity.n + 1

        # sector sizes
        self.n0 = self.m
        self.n1 = 2 * self.m
        self.n2 = self.m

        self.dim = self.n0 + self.n1 + self.n2

        # index slices
        self.P0 = slice(0, self.n0)
        self.P1 = slice(self.n0, self.n0 + self.n1)
        self.P2 = slice(self.n0 + self.n1, self.dim)

        # build master matrix immediately
        self.L, self.blocks = self._build_master()

    # -------------------------------------------------
    # Build master equation matrix
    # -------------------------------------------------
    def _build_master(self):
        """
        Construct master equation matrix L from Γ_tot.
        """

        L = np.zeros((self.dim, self.dim), dtype=float)

        # --------------------------------------------------------- 
        
        """
        note that the rate i → f is given by Γ_{if} = rate (i,f) but in the rate 
        equation writes:
        
            dP_i/dt = - (L P)_i + L_{ii} P_i
                    = - \sum_f L_{if} P_f + L_{ii} P_i

        by comparing one gets L = Γ.T such that

            L_{if} = Γ_{fi}            for off-diagonal
            L_{ii} = \sum_f Γ_{if}     for diagonal
        
        """
        
        # Incoming rates: L_{fi} = Γ(i → f)
        # ---------------------------------------------------------
        
        L[:, :] = self.Γ_full.T

        # ---------------------------------------------------------
        # Fix diagonal elements:
        # L[i,i] = - sum_{f ≠ i} Γ(i → f)
        # ---------------------------------------------------------
        for i in range(self.dim):
            outgoing = np.sum(self.Γ_full.T[:, i])
            L[i, i] = -outgoing

        # ---------------------------------------------------------
        # Extract blocks (exact sector block structure)
        # ---------------------------------------------------------
        A = L[self.P0, self.P0]
        B = L[self.P0, self.P1]
        D = L[self.P1, self.P0]
        E = L[self.P1, self.P1]
        F = L[self.P1, self.P2]
        H = L[self.P2, self.P1]
        I = L[self.P2, self.P2]

        blocks = {
            "A": A, "B": B, "D": D,
            "E": E, "F": F,
            "H": H, "I": I
        }

        return L, blocks

    # -------------------------------------------------
    # Solve steady state
    # -------------------------------------------------
    def steady_solver(self):
        evals, evecs = np.linalg.eig(self.L)
        idx = np.argmin(np.abs(evals))
        P = np.real(evecs[:, idx])
        P /= np.sum(P)
        return P


# ======================================================================
# 5a. `TransportCalculator` — current I from steady-state populations
# ======================================================================
class TransportCalculator:
    """
    Computes transport observables (currently current)
    from sector-ordered rate matrices and steady-state probabilities.
    """

    def __init__(self, cavity: CavityParameters, leads: LeadParameters):
        self.cavity = cavity
        self.leads = leads

        self.n_ph = cavity.n
        self.n0 = self.n_ph + 1
        self.n1 = 2 * (self.n_ph + 1)
        self.n2 = self.n_ph + 1
        self.dim = self.n0 + self.n1 + self.n2

        # sector slices (consistent everywhere in your codebase)
        self.i0 = slice(0, self.n0)
        self.i1 = slice(self.n0, self.n0 + self.n1)
        self.i2 = slice(self.n0 + self.n1, self.dim)

    # -------------------------------------------------
    # Current from one lead
    # -------------------------------------------------
    def compute_current(self, Γ_alpha, P):
        """
        Compute current J^(alpha) from lead alpha.

        Parameters
        ----------
        Γ_alpha : ndarray (dim x dim)
            Transition-rate matrix for one lead (Γ_L or Γ_R),
            with Γ[f,i] = rate i -> f.

        P : ndarray (dim,)
            Steady-state probability vector.

        Returns
        -------
        J : float
            Current from lead alpha.
        """

        if Γ_alpha.shape != (self.dim, self.dim):
            raise ValueError("Gamma matrix has wrong dimension.")
        if P.shape[0] != self.dim:
            raise ValueError("Probability vector has wrong dimension.")

        # Sector probability vectors
        P0 = P[self.i0]
        P1 = P[self.i1]
        P2 = P[self.i2]

        # Extract sector blocks
        G01 = Γ_alpha[self.i0, self.i1]   # 0 -> 1
        G10 = Γ_alpha[self.i1, self.i0]   # 1 -> 0
        G12 = Γ_alpha[self.i1, self.i2]   # 1 -> 2
        G21 = Γ_alpha[self.i2, self.i1]   # 2 -> 1

        # -------------------------------------------------
        # Vectorized sector-block current formula
        # -------------------------------------------------

        # term1 = sum_i P0[i] * sum_j Γ(i→j in 1)
        term1 = float(P0.dot(G01.sum(axis=1)))

        # term2 = sum_j P1[j] * sum_i (Γ(j→i in 2) - Γ(j→i in 0))
        term2 = float(P1.dot((G12 - G10).sum(axis=1)))

        # term3 = sum_i P2[i] * sum_j Γ(i→j in 1)
        term3 = float(P2.dot(G21.sum(axis=1)))

        # Symmetric normalization (your convention)
        γL = self.leads.gamma_L
        γR = self.leads.gamma_R
        norm = γL * γR / (γL + γR)

        J = (term1 + term2 - term3) / norm

        return J

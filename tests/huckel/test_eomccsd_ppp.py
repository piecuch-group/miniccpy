from miniccpy.models.huckel import ppp_hamiltonian
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

if __name__ == "__main__":

    n_sites = 6
    flag_cyclic = True
    alpha = 0.0 
    beta = -2.4
    gamma = 10.84
    r = 1.4

    # Obtain PPP model hamiltonian
    z, g, fock, o, v, e_hf = ppp_hamiltonian(n_sites, flag_cyclic, alpha, beta, gamma, r)

    # Run CC calculation
    T, E_corr = run_cc_calc(fock, g, o, v, method="ccsd", energy_shift=0.0)
    # Compute HBar
    H1, H2 = get_hbar(T, fock, g, o, v, method="ccsd")
    # Run CIS-type initial guess
    R, omega_guess = run_guess(H1, H2, o, v, 5, method="cis", mult=1)
    # Run the EOMCC calculation
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomccsd", state_index=[0, 1, 2, 3, 4], max_size=20)

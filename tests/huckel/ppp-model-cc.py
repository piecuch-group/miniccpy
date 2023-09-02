from miniccpy.models.huckel import ppp_hamiltonian
from miniccpy.driver import run_cc_calc

if __name__ == "__main__":

    n_sites = 10
    flag_cyclic = False
    alpha = 0.0 
    beta = -2.4
    gamma = 10.84
    r = 1.4

    # Obtain PPP model hamiltonian
    z, g, fock, o, v, e_hf = ppp_hamiltonian(n_sites, flag_cyclic, alpha, beta, gamma, r)

    # Run CC calculation
    T, E_corr = run_cc_calc(fock, g, o, v, method="ccsd")

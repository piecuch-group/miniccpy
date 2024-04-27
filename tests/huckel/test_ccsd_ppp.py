from miniccpy.models.huckel import ppp_hamiltonian
from miniccpy.driver import run_mpn_calc, run_cc_calc
from miniccpy.constants import hartree_to_eV 

if __name__ == "__main__":

    n_sites = 6
    flag_cyclic = True
    alpha = 0.0 
    beta = -0.0001
    gamma = 10.84
    r = 1.4

    # Obtain PPP model hamiltonian
    z, g, fock, o, v, e_hf = ppp_hamiltonian(n_sites, flag_cyclic, alpha, beta, gamma, r, hubbard=False)

    e_mp2 = run_mpn_calc(fock, g, o, v, method="mp2")

    # Run CC calculation
    T, E_corr = run_cc_calc(fock, g, o, v, method="ccsd", energy_shift=0.0)

    # Compute correlation energy per electron in eV
    print(f"Correlation energy per electron = {E_corr / n_sites * hartree_to_eV} eV")


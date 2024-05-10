import numpy as np
from pathlib import Path
from miniccpy.driver import run_scf, run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar
from miniccpy.pspace import get_active_4h2p_pspace

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

#basis = '6-31g'
#nfrozen = 2
# Define molecule geometry and basis set
#geom = [['F', (0.0, 0.0, -2.66816)], 
#        ['F', (0.0, 0.0,  2.66816)]]

#fock, g, e_hf, o, v, orbsym = run_scf(geom, basis, nfrozen, unit="Bohr", symmetry="D2H", cartesian=False, charge=-2, return_orbsym=True)

fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/f2-631g.FCIDUMP", 20, 18, nfrozen=2)

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')
H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

nroot = 4
R, omega_guess = run_guess(H1, H2, o, v, nroot, method="dipcis")

no, nu = fock[o, v].shape
r3_excitations = get_active_4h2p_pspace(no, nu, nacto=2)
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="dipeom4_p", state_index=[0, 3], r3_excitations=r3_excitations, max_size=80)

#
# Check the results
#
expected_vee = [-0.1347915045, -0.1348578163]
for i, vee in enumerate(expected_vee):
    assert np.allclose(omega[i], vee, atol=1.0e-07)

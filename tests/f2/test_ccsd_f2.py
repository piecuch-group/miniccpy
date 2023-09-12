from miniccpy.driver import run_scf, run_cc_calc

basis = 'cc-pvdz'
nfrozen = 2

geom = [['F', (0.0, 0.0,  2.66816)],
        ['F', (0.0, 0.0, -2.66816)]] 

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, cartesian=True, unit="Bohr")

T, E_corr = run_cc_calc(fock, g, o, v, method='ccsd')








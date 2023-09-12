from miniccpy.driver import run_scf, run_cc_calc

basis = 'cc-pvdz'
nfrozen = 0
Re = 5

geom = [['H', (-Re, -Re, 0.000)], 
        ['H', (-Re,  Re, 0.000)], 
        ['H', ( Re, -Re, 0.000)], 
        ['H', ( Re,  Re, 0.000)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200)

T, E_corr = run_cc_calc(fock, g, o, v, method='ccsd', maxit=80, energy_shift=0.3, use_quasi=False)








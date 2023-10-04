from miniccpy.driver import run_scf, run_mpn_calc

basis = 'cc-pvdz'
nfrozen = 0
Re = 1.0

geom = [['H', (-Re, -Re, 0.000)], 
        ['H', (-Re,  Re, 0.000)], 
        ['H', ( Re, -Re, 0.000)], 
        ['H', ( Re,  Re, 0.000)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200)

E_corr = run_mpn_calc(fock, g, o, v, method='mp2')








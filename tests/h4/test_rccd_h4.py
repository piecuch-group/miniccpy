from miniccpy.driver import run_scf, run_cc_calc

basis = 'cc-pvdz'
nfrozen = 0
Re = 1.0

geom = [['H', (-Re, -Re, 0.000)], 
        ['H', (-Re,  Re, 0.000)], 
        ['H', ( Re, -Re, 0.000)], 
        ['H', ( Re,  Re, 0.000)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200, rhf=True)

T, E_corr = run_cc_calc(fock, g, o, v, method='rccd', maxit=80)








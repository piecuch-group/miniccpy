# set allow numpy built with MKL to consume more threads for tensordot
#import os
#os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

from miniccpy.driver import run_scf, run_cc_calc

basis = 'ccpvdz'
nfrozen = 2

# Define molecule geometry and basis set
geom = [['F', (0.0, 0.0, -2.66816)], 
        ['F', (0.0, 0.0,  2.66816)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, multiplicity=2, charge=1, maxit=200)

T, E_corr = run_cc_calc(fock, g, o, v, method='ccsd', out_of_core=False, use_quasi=True)


from pyscf import gto, scf, cc
mol = gto.Mole()

mol.build(
        atom=geom,
        basis=basis,
        charge=1,
        spin=1,
        cart=False,
        unit='Bohr',
        symmetry=True,
)
mf = scf.ROHF(mol)
mf.kernel()

mycc = cc.UCCSD(mf, frozen=2)
mycc.kernel()









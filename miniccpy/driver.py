import time
import numpy as np
from importlib import import_module

from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))

__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
MODULES = [module for module in __all__]


def run_scf(geometry, basis, nfrozen=0, multiplicity=1, charge=0):
    """Run the ROHF calculation using PySCF and obtain the molecular
    orbital integrals in normal-ordered form as well as the occupied/
    unoccupied slicing arrays for correlated calculations."""
    from pyscf import gto, scf
    from miniccpy.printing import print_system_information
    from miniccpy.integrals import get_integrals_from_pyscf
    from miniccpy.energy import hf_energy

    mol = gto.Mole()

    mol.build(
        atom=geometry,
        basis=basis,
        charge=charge,
        spin=multiplicity-1,
        cart=False,
        unit='Bohr',
        symmetry=True,
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    # 1-, 2-electron spinorbital integrals in physics notation
    e1int, e2int, fock, e_hf, nuclear_repulsion = get_integrals_from_pyscf(mf)

    corr_occ = slice(2 * nfrozen, mf.mol.nelectron)
    corr_unocc = slice(mf.mol.nelectron, 2 * mf.mo_coeff.shape[1])

    print_system_information(mf, nfrozen, e_hf)

    return fock, e2int, e_hf, corr_occ, corr_unocc

def run_cc_calc(fock, g, o, v, method, maxit=80, convergence=1.0e-07, diis_size=6, n_start_diis=3, energy_shift=0.0, out_of_core=False):
    """Run the ground-state CC calculation specified by `method`."""

    # check if requested CC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')

    # Turn off DIIS for small systems; it becomes singular!
    if fock.shape[0] <= 4: 
        print("Turning off DIIS acceleration for small system")
        diis_size = 1000 

    tic = time.time()
    T, e_corr = calculation(fock, g, o, v, maxit, convergence, diis_size, n_start_diis, energy_shift, out_of_core)
    toc = time.time()

    minutes, seconds = divmod(toc - tic, 60)

    print("")
    print("    CC Correlation Energy: {: 20.12f}".format(e_corr))
    print("")
    print("CC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))

    return T, e_corr

def get_hbar(T, fock, g, o, v, method):
    """Obtain the similarity-transformed Hamiltonian Hbar corresponding
    to the level of ground-state CC theory specified by `method`."""

    # import the specific CC method module and get its update function
    mod = import_module("miniccpy.hbar")
    hbar_builder = getattr(mod, 'build_hbar_'+method.lower())

    H1, H2 = hbar_builder(T, fock, g, o, v)

    return H1, H2

def run_eomcc_calc(T, fock, H1, H2, o, v, nroot, method, maxit=80, convergence=1.0e-07):
    """Run the excited-state EOMCC calculation specified by `method`.
    Currently, this module only supports CIS initial guesses."""

    from miniccpy.initial_guess import get_initial_guess

    # check if requested EOMCC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')

    # get the initial guess
    R0, omega0 = get_initial_guess(H1, H2, o, v, nroot) 

    R = [0 for i in range(nroot)]
    omega = [0 for i in range(nroot)]
    r0 = [0 for i in range(nroot)]
    for n in range(nroot):
        tic = time.time()
        R[n], omega[n], r0[n] = calculation(R0[:, n], T, omega0[n], H1, H2, o, v, maxit, convergence)
        toc = time.time()

        minutes, seconds = divmod(toc - tic, 60)

        print("")
        print("    EOMCC Excitation Energy: {: 20.12f}".format(omega[n]))
        print("")
        print("EOMCC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))

    return R, omega, r0




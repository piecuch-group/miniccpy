import time
import numpy as np
from importlib import import_module

from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))

__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
MODULES = [module for module in __all__]

def run_scf_gamess(fcidump, nelectron, norbitals, nfrozen=0):
    """Obtain the mean-field solution from GAMESS FCIDUMP file and 
    return the necessary objects, including MO integrals and correlated
    slicing arrays for the CC calculation"""
    from miniccpy.integrals import get_integrals_from_gamess
    from miniccpy.printing import print_custom_system_information

    # 1-, 2-electron spinorbital integrals in physics notation
    e1int, e2int, fock, e_hf, nuclear_repulsion = get_integrals_from_gamess(fcidump, nelectron, norbitals)

    corr_occ = slice(2 * nfrozen, nelectron)
    corr_unocc = slice(nelectron, 2 * norbitals)

    print_custom_system_information(fock, nelectron, nfrozen, e_hf)

    return fock, e2int, e_hf, corr_occ, corr_unocc

def run_scf(geometry, basis, nfrozen=0, multiplicity=1, charge=0, 
            maxit=200, level_shift=0.0, damp=0.0, convergence=1.0e-10, cartesian=False, unit="Bohr"):
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
        cart=cartesian,
        unit=unit,
        symmetry=True,
    )
    mf = scf.ROHF(mol)
    # Put in SCF options for PySCF
    mf.level_shift = level_shift
    mf.damp = damp
    mf.max_cycle = maxit
    mf.conv_tol = convergence
    mf.kernel()

    # 1-, 2-electron spinorbital integrals in physics notation
    e1int, e2int, fock, e_hf, nuclear_repulsion = get_integrals_from_pyscf(mf)

    corr_occ = slice(2 * nfrozen, mf.mol.nelectron)
    corr_unocc = slice(mf.mol.nelectron, 2 * mf.mo_coeff.shape[1])

    print_system_information(mf, nfrozen, e_hf)

    return fock, e2int, e_hf, corr_occ, corr_unocc

def run_cc_calc(fock, g, o, v, method, 
                maxit=80, convergence=1.0e-07, shift=0.0, diis_size=6, n_start_diis=3, energy_shift=0.0, out_of_core=False, use_quasi=False):
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
    T, e_corr = calculation(fock, g, o, v, maxit, convergence, shift, diis_size, n_start_diis, energy_shift, out_of_core, use_quasi)
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

def run_guess(H1, H2, o, v, nroot, method="cis"):
    """Run the CIS initial guess to obtain starting vectors for the EOMCC iterations."""
    from miniccpy.initial_guess import cis_guess, eacis_guess

    no, nu = H1[o, v].shape
    nroot = min(nroot, no * nu)

    # get the initial guess
    if method == "cis":
        R0, omega0 = cis_guess(H1, H2, o, v, nroot)
    elif method == "eacis":
        R0, omega0 = eacis_guess(H1, H2, o, v, nroot)
    
    print("Initial guess energies:")
    for i, e in enumerate(omega0):
        print("  Guess root ", i + 1, " = ", np.real(e))
    print("")

    return np.real(R0), np.real(omega0)

def run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method, state_index, maxit=80, convergence=1.0e-07):
    """Run the excited-state EOMCC calculation specified by `method`.
    Currently, this module only supports CIS initial guesses."""


    # check if requested EOMCC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')

    nroot = len(state_index)

    R = [0 for i in range(nroot)]
    omega = [0 for i in range(nroot)]
    r0 = [0 for i in range(nroot)]
    for n in range(nroot):
        tic = time.time()
        R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n] - 1], T, omega0[state_index[n] - 1], H1, H2, o, v, maxit, convergence)
        toc = time.time()

        minutes, seconds = divmod(toc - tic, 60)

        print("")
        print("    EOMCC Excitation Energy: {: 20.12f}".format(omega[n]))
        print("    r0 = {: 20.12f}".format(r0[n]))
        print("    REL = {: 20.12f}".format(rel))
        print("")
        print("EOMCC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))

    return R, omega, r0




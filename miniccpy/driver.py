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
            maxit=200, level_shift=0.0, damp=0.0, convergence=1.0e-10,
            symmetry=None, cartesian=False, unit="Bohr"):
    """Run the ROHF calculation using PySCF and obtain the molecular
    orbital integrals in normal-ordered form as well as the occupied/
    unoccupied slicing arrays for correlated calculations."""
    from pyscf import gto, scf
    from miniccpy.printing import print_system_information
    from miniccpy.integrals import get_integrals_from_pyscf
    from miniccpy.energy import hf_energy

    if symmetry is None:
        point_group = True
    else:
        point_group = symmetry

    mol = gto.Mole()
    mol.build(
        atom=geometry,
        basis=basis,
        charge=charge,
        spin=multiplicity-1,
        cart=cartesian,
        unit=unit,
        symmetry=point_group,
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

def run_mpn_calc(fock, g, o, v, method):
    """Compute the Moller-Plesett energy correction specified
    by `method`."""
    # check if requested MBPT calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')

    # Run the MBPT calculation to obtain the correlation energy
    tic = time.time()
    e_corr = calculation(fock, g, o, v)
    toc = time.time()
    minutes, seconds = divmod(toc - tic, 60)
    print("")
    print("    MPn Correlation Energy: {: 20.12f}".format(e_corr))
    print("")
    print("    MPn calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
    print("")

    return e_corr


def run_cc_calc(fock, g, o, v, method, 
                maxit=80, convergence=1.0e-07, energy_shift=0.0, diis_size=6, n_start_diis=3, out_of_core=False, use_quasi=False):
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
    T, e_corr = calculation(fock, g, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core, use_quasi)
    toc = time.time()

    minutes, seconds = divmod(toc - tic, 60)

    print("")
    print("    CC Correlation Energy: {: 20.12f}".format(e_corr))
    print("")
    print("    CC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
    print("")

    return T, e_corr

def get_hbar(T, fock, g, o, v, method):
    """Obtain the similarity-transformed Hamiltonian Hbar corresponding
    to the level of ground-state CC theory specified by `method`."""

    # import the specific CC method module and get its update function
    mod = import_module("miniccpy.hbar")
    hbar_builder = getattr(mod, 'build_hbar_'+method.lower())

    H1, H2 = hbar_builder(T, fock, g, o, v)

    return H1, H2

def run_guess(H1, H2, o, v, nroot, method, nacto=0, nactu=0, print_threshold=0.025, mult=-1):
    """Run the CIS initial guess to obtain starting vectors for the EOMCC iterations."""
    from miniccpy.initial_guess import cis_guess, cisd_guess, eacis_guess, ipcis_guess
    from miniccpy.printing import print_cis_vector, print_cisd_vector, print_1p_vector, print_1h_vector

    no, nu = H1[o, v].shape

    # get the initial guess
    if method == "cisd":
        nroot = min(nroot, no * nu)
        R0, omega0 = cisd_guess(H1, H2, o, v, nroot, nacto, nactu, mult)
    elif method == "cis":
        nroot = min(nroot, no * nu)
        R0, omega0 = cis_guess(H1, H2, o, v, nroot, mult)
    elif method == "eacis":
        nroot = min(nroot, nu)
        R0, omega0 = eacis_guess(H1, H2, o, v, nroot)
    elif method == "ipcis":
        nroot = min(nroot, no)
        R0, omega0 = ipcis_guess(H1, H2, o, v, nroot)
    
    print("    Initial Guess Vectors:")
    print("    -----------------------")
    for i, e in enumerate(omega0):
        print("    Root ", i + 1)
        print("    Energy = ", np.real(e))
        print("    Largest Amplitudes:")
        if method == "cis":
            print_cis_vector(np.real(R0[:, i].reshape(nu, no)), print_threshold=print_threshold)
        elif method == "cisd":
            print_cisd_vector(np.real(R0[:no*nu, i].reshape(nu, no)), np.real(R0[no*nu:, i].reshape(nu, nu, no, no)), print_threshold=print_threshold)
        elif method == "eacis":
            print_1p_vector(np.real(R0[:, i]), no, print_threshold=print_threshold)
        elif method == "ipcis":
            print_1h_vector(np.real(R0[:, i]), nu, print_threshold=print_threshold)
        print("")
    print("")

    return np.real(R0), np.real(omega0)

def run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method, state_index, fock=None, g=None, maxit=80, convergence=1.0e-07, max_size=20, diis_size=6, do_diis=True, denom_type="fock"):
    """Run the IP-/EA- or EE-EOMCC calculation specified by `method`.
    Currently, this module only supports CIS-type initial guesses."""


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
        print(f"    Solving for state #{state_index[n]}")
        tic = time.time()
        # Note: EOMCC3 methods have a difference function call due to needing fock and g matrices
        if method.lower() == "eomcc3": # Folded EOMCC3 model using excited-state DIIS algorithm
            R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n]], T, omega0[state_index[n]], fock, g, H1, H2, o, v, maxit, convergence, diis_size=diis_size, do_diis=do_diis, denom_type=denom_type)
        elif method.lower() == "dreomcc3": # Folded dressed EOMCC3 model using excited-state DIIS algorithm
            R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n]], T, omega0[state_index[n]], H1, H2, o, v, maxit, convergence, diis_size=diis_size, do_diis=do_diis)
        elif method.lower() == "eomcc3-lin": # Linear EOMCC3 model using conventional Davidson diagonalization
            R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n]], T, omega0[state_index[n]], fock, g, H1, H2, o, v, maxit, convergence, max_size=max_size, diis_size=diis_size, do_diis=do_diis)
        else: # All other EOMCC calculations using conventional Davidson
            R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n]], T, omega0[state_index[n]], H1, H2, o, v, maxit, convergence, max_size=max_size)
        toc = time.time()

        minutes, seconds = divmod(toc - tic, 60)

        print("")
        print("    EOMCC Excitation Energy: {: 20.12f}".format(omega[n]))
        print("    r0 = {: 20.12f}".format(r0[n]))
        print("    REL = {: 20.12f}".format(rel))
        print("")
        print("    EOMCC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
        print("")

    return R, omega, r0




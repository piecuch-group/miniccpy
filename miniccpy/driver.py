import time
import numpy as np
from importlib import import_module
from os.path import dirname, basename, isfile, join
import glob
from miniccpy.utilities import get_memory_usage

# Obtain all modules in Miniccpy
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
MODULES = [module for module in __all__]
# Manually specify those modules that are RHF non-orthogonally spin-adapted codes
RHF_MODULES = ["rlccd", "rccd", "rccsd", "rccsdt", "left_rccsd", "left_eomrccsd", "eomrccsd", "rcc3", "rccsdt", "eomrccsdt"]

# amplitude printing threshold
PRINT_THRESH = 0.025

def run_scf_gamess(fcidump, nelectron, norbitals, nfrozen=0, rhf=False):
    """Obtain the mean-field solution from GAMESS FCIDUMP file and 
    return the necessary objects, including MO integrals and correlated
    slicing arrays for the CC calculation"""
    from miniccpy.integrals import get_integrals_from_gamess
    from miniccpy.printing import print_custom_system_information, print_custom_system_information_rhf

    # 1-, 2-electron spinorbital integrals in physics notation
    e1int, e2int, fock, e_hf, nuclear_repulsion = get_integrals_from_gamess(fcidump, nelectron, norbitals, rhf=rhf)

    if rhf:
        corr_occ = slice(nfrozen, int(nelectron / 2))
        corr_unocc = slice(int(nelectron / 2), norbitals)
        print_custom_system_information_rhf(fock, nelectron, nfrozen, e_hf)
    else:
        corr_occ = slice(2 * nfrozen, nelectron)
        corr_unocc = slice(nelectron, 2 * norbitals)
        print_custom_system_information(fock, nelectron, nfrozen, e_hf)

    return fock, e2int, e_hf, corr_occ, corr_unocc

def run_scf(geometry, basis, nfrozen=0, multiplicity=1, charge=0, 
            maxit=200, level_shift=0.0, damp=0.0, convergence=1.0e-10,
            symmetry=None, cartesian=False, unit="Bohr", uhf=False, rhf=False,
            return_orbsym=False, x2c=False, multipole=0):
    """Run the ROHF calculation using PySCF and obtain the molecular
    orbital integrals in normal-ordered form as well as the occupied/
    unoccupied slicing arrays for correlated calculations."""
    from pyscf import gto, scf, symm
    from miniccpy.printing import print_system_information, print_custom_system_information
    from miniccpy.integrals import get_integrals_from_pyscf, get_integrals_from_pyscf_uhf, get_integrals_from_pyscf_rhf
    from miniccpy.multipoles import get_multipole_integrals 

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
    if uhf:
        if x2c:
            mf = scf.UHF(mol).x2c()
        else:
            mf = scf.UHF(mol)
    elif rhf:
        if x2c:
            mf = scf.RHF(mol).x2c()
        else:
            mf = scf.RHF(mol)
    else:
        if x2c:
            mf = scf.ROHF(mol).x2c()
        else:
            mf = scf.ROHF(mol)
    # Put in SCF options for PySCF
    mf.level_shift = level_shift
    mf.damp = damp
    mf.max_cycle = maxit
    mf.conv_tol = convergence
    mf.kernel()

    # Get list of orbital symmetry labels
    orbsym = [x.upper() for x in symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)]
    # make this into spinorbital labels
    sporbsym = []
    for p in range(2 * len(orbsym)):
        if p % 2 == 0:
            sporbsym.append(orbsym[p // 2])
        else:
            sporbsym.append(orbsym[(p - 1) // 2])

    # 1-, 2-electron spinorbital integrals in physics notation
    if uhf:
        e1int, e2int, fock, e_hf, nuclear_repulsion = get_integrals_from_pyscf_uhf(mf)
    elif rhf:
        e1int, e2int, fock, e_hf, nuclear_repulsion = get_integrals_from_pyscf_rhf(mf)
        corr_occ = slice(nfrozen, int(mf.mol.nelectron / 2))
        corr_unocc = slice(int(mf.mol.nelectron / 2), e1int.shape[0])
    else:
        e1int, e2int, fock, e_hf, nuclear_repulsion = get_integrals_from_pyscf(mf)
        corr_occ = slice(2 * nfrozen, mf.mol.nelectron)
        corr_unocc = slice(mf.mol.nelectron, e1int.shape[0])

    if uhf:
        print_custom_system_information(fock, mf.mol.nelectron, nfrozen, e_hf)
    else:
        print_system_information(mf, nfrozen, e_hf)

    if multipole != 0:
        mu = get_multipole_integrals(multipole, mol, mf) 
        if return_orbsym:
            return fock, e2int, e_hf, corr_occ, corr_unocc, mu, sporbsym[2*nfrozen:]
        else:
            return fock, e2int, e_hf, corr_occ, corr_unocc, mu
    else:
        if return_orbsym:
            return fock, e2int, e_hf, corr_occ, corr_unocc, sporbsym[2*nfrozen:]
        else:
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
    print(f"    Memory usage: {get_memory_usage()} MB")
    print("")
    return e_corr

def run_cmx_calc(T, L, Ecorr, H1, H2, o, v, method):
    """Compute the CMX energy correction specified
    by `method`."""
    # check if requested MBPT calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')

    # Run the CMX calculation to obtain the correction to the CC correlation energy
    tic = time.time()
    delta_corr = calculation(T, L, Ecorr, H1, H2, o, v)
    toc = time.time()
    minutes, seconds = divmod(toc - tic, 60)
    print("")
    print("    CMX Correction Energy: {: 20.12f}".format(delta_corr))
    print("    Total Corrlation Energy: {: 20.12f}".format(Ecorr + delta_corr))
    print("")
    print("    CMX calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
    print(f"    Memory usage: {get_memory_usage()} MB")
    print("")
    return delta_corr

def run_cc_calc(fock, g, o, v, method, maxit=80, convergence=1.0e-07, energy_shift=0.0, diis_size=6, n_start_diis=0, out_of_core=False, use_quasi=False, t3_excitations=None):
    """Run the ground-state CC calculation specified by `method`."""
    from miniccpy.printing import print_amplitudes

    # check if requested CC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    if method in RHF_MODULES:
        flag_rhf = True
    else:
        flag_rhf = False
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')

    # Turn off DIIS for small systems; it becomes singular!
    if fock.shape[0] <= 4: 
        print("Turning off DIIS acceleration for small system")
        diis_size = 1000 

    tic = time.time()
    if t3_excitations is not None:
        T, e_corr = calculation(fock, g, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core, use_quasi, t3_excitations=t3_excitations)
    else:
        T, e_corr = calculation(fock, g, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core, use_quasi)
    toc = time.time()

    minutes, seconds = divmod(toc - tic, 60)

    print("")
    print("    CC Correlation Energy: {: 20.12f}".format(e_corr))
    print("")
    print("    Largest Singly and Doubly Excited Amplitudes")
    print("    --------------------------------------------")
    print_amplitudes(T[0], T[1], PRINT_THRESH, rhf=flag_rhf)
    print("")
    print("    CC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
    print(f"    Memory usage: {get_memory_usage()} MB")
    print("")

    return T, e_corr

def run_leftcc_calc(T, fock, H1, H2, o, v, method, maxit=80, convergence=1.0e-07, energy_shift=0.0, diis_size=6, n_start_diis=0, out_of_core=False, davidson=False, g=None):
    """Run the ground-state left-CC calculation specified by `method`."""
    from miniccpy.printing import print_amplitudes

    # check if requested left-CC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    if method in RHF_MODULES:
        flag_rhf = True
    else:
        flag_rhf = False
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')
    # Turn off DIIS for small systems; it becomes singular!
    if H1.shape[0] <= 4: 
        print("Turning off DIIS acceleration for small system")
        diis_size = 1000 
    # Run the linear equation solver
    tic = time.time()
    if method in ["left_cc3", "left_cc3-full"]:
        L, omega = calculation(T, fock, g, H1, H2, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core)
    else:
        L, omega = calculation(T, fock, H1, H2, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core)
    toc = time.time()

    minutes, seconds = divmod(toc - tic, 60)
    print("")
    print("    Left-CC Excitation Energy: {: 20.12f}".format(omega))
    print("")
    print("    Largest Singly and Doubly Excited Amplitudes")
    print("    --------------------------------------------")
    print_amplitudes(L[0], L[1], PRINT_THRESH, rhf=flag_rhf)
    print("")
    print("    Left-CC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
    print(f"    Memory usage: {get_memory_usage()} MB")
    print("")
    return L

def get_hbar(T, fock, g, o, v, method, **kwargs):
    """Obtain the similarity-transformed Hamiltonian Hbar corresponding
    to the level of ground-state CC theory specified by `method`."""

    # import the specific CC method module and get its update function
    mod = import_module("miniccpy.hbar")
    hbar_builder = getattr(mod, 'build_hbar_'+method.lower())

    H1, H2 = hbar_builder(T, fock, g, o, v, **kwargs)

    return H1, H2

def run_guess(H1, H2, o, v, nroot, method, nacto=0, nactu=0, print_threshold=PRINT_THRESH, mult=-1, cvsmin=-1, cvsmax=-1):
    """Run the CIS initial guess to obtain starting vectors for the EOMCC iterations."""
    from miniccpy.initial_guess import cis_guess, rcis_guess, rcisd_guess, cisd_guess, eacis_guess, ipcis_guess, deacis_guess, dipcis_guess, dipcis_cvs_guess, dipcisd_guess, dipcisd_cvs_guess
    from miniccpy.printing import print_cis_vector, print_rcis_vector, print_rcisd_vector, print_cisd_vector, print_1p_vector, print_1h_vector, print_2p_vector, print_2h_vector, print_dip_amplitudes

    no, nu = H1[o, v].shape

    # get the initial guess
    tic = time.perf_counter()
    if method == "cisd":
        nroot = min(nroot, no * nu + int(nacto*(nacto - 1)/2 * nactu*(nactu - 1)/2))
        R0, omega0 = cisd_guess(H1, H2, o, v, nroot, nacto, nactu, mult)
    elif method == "cis":
        nroot = min(nroot, no * nu)
        R0, omega0 = cis_guess(H1, H2, o, v, nroot, mult)
    elif method == "rcis":
        nroot = min(nroot, no * nu)
        R0, omega0 = rcis_guess(H1, H2, o, v, nroot, mult=1)
    elif method == "rcisd":
        nroot = min(nroot, no * nu + int(nacto*(nacto - 1)/2 * nactu*(nactu - 1)/2))
        R0, omega0 = rcisd_guess(H1, H2, o, v, nroot, nacto, nactu, mult=1)
    elif method == "eacis":
        nroot = min(nroot, nu)
        R0, omega0 = eacis_guess(H1, H2, o, v, nroot)
    elif method == "deacis":
        nroot = min(nroot, nu**2)
        R0, omega0 = deacis_guess(H1, H2, o, v, nroot, nactu)
    elif method == "ipcis":
        nroot = min(nroot, no)
        R0, omega0 = ipcis_guess(H1, H2, o, v, nroot)
    elif method == "dipcis":
        nroot = min(nroot, no**2)
        if cvsmin != -1 and cvsmax != -1:
            R0, omega0 = dipcis_cvs_guess(H1, H2, o, v, nroot, cvsmin, cvsmax)
        else:
            R0, omega0 = dipcis_guess(H1, H2, o, v, nroot)
    elif method == "dipcisd":
        nroot = min(nroot, int(no*(no - 1)/2 + nacto*(nacto - 1)*(nacto - 2)/6 * nactu))
        if cvsmin != -1 and cvsmax != -1:
            R0, omega0 = dipcisd_cvs_guess(H1, H2, o, v, nroot, cvsmin, cvsmax, nacto, nactu)
        else:
            R0, omega0 = dipcisd_guess(H1, H2, o, v, nroot, nacto, nactu)

    toc = time.perf_counter()
    minutes, seconds = divmod(toc - tic, 60)

    # Convert initial vector to real
    R0 = np.real(R0)

    print("    Initial Guess Vectors:")
    print("    -----------------------")
    print("    Guess calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
    print(f"    Memory usage: {get_memory_usage()} MB\n")
    for i, e in enumerate(omega0):
        print("    Root ", i + 1)
        print("    Energy = ", np.real(e))
        print("    Largest Amplitudes:")
        if method == "cis":
            print_cis_vector(R0[:, i].reshape(nu, no), print_threshold=print_threshold)
        elif method == "rcis":
            print_rcis_vector(R0[:, i].reshape(nu, no), print_threshold=print_threshold)
        elif method == "rcisd":
            print_rcisd_vector(R0[:no*nu, i].reshape(nu, no), R0[no*nu:, i].reshape(nu, nu, no, no), print_threshold=print_threshold)
        elif method == "cisd":
            print_cisd_vector(R0[:no*nu, i].reshape(nu, no), R0[no*nu:, i].reshape(nu, nu, no, no), print_threshold=print_threshold)
        elif method == "eacis":
            print_1p_vector(R0[:, i], no, print_threshold=print_threshold)
        elif method == "ipcis":
            print_1h_vector(R0[:, i], nu, print_threshold=print_threshold)
        elif method == "deacis":
            print_2p_vector(R0[:nu**2, i].reshape(nu, nu), no, print_threshold=print_threshold)
        elif method == "dipcis":
            print_2h_vector(R0[:no**2, i].reshape(no, no), nu, print_threshold=print_threshold)
        elif method == "dipcisd":
           print_dip_amplitudes(R0[:no**2, i].reshape(no, no), R0[no**2:, i].reshape(no, no, nu, no), print_threshold=print_threshold)
        print("")
    print("")

    return np.real(R0), np.real(omega0)

def run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method, state_index, fock=None, g=None, maxit=80, convergence=1.0e-07, max_size=20, diis_size=6,
                   do_diis=True, r3_excitations=None, out_of_core=False, cvsmin=-1, cvsmax=-1):
    """Run the IP-/EA- or EE-EOMCC calculation specified by `method`.
    Currently, this module only supports CIS-type initial guesses."""
    from miniccpy.printing import print_amplitudes, print_dip_amplitudes

    # check if requested EOMCC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    if method in RHF_MODULES:
        flag_rhf = True
    else:
        flag_rhf = False
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
        if method.lower() == "eomcc3" or method.lower() == "eomrcc3": # Folded EOMCC3 model
            R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n]], T, omega0[state_index[n]], fock, g, H1, H2, o, v, maxit, convergence, diis_size=diis_size, do_diis=do_diis)
        elif method.lower() == "eomccsdta" or method.lower() == "eomrccsdta":
            R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n]], T, omega0[state_index[n]], fock, g, H1, H2, o, v, maxit, convergence, max_size=max_size, out_of_core=out_of_core)
        elif method.lower() == "dreomcc3": # Folded dressed EOMCC3 model using excited-state DIIS algorithm
            R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n]], T, omega0[state_index[n]], H1, H2, o, v, maxit, convergence, diis_size=diis_size, do_diis=do_diis)
        elif method.lower() == "eomcc3-lin": # Linear EOMCC3 model using conventional Davidson diagonalization
            R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n]], T, omega0[state_index[n]], fock, g, H1, H2, o, v, maxit, convergence, max_size=max_size)
        elif method.lower() == "dipeom4_star_p": # Approximate DIP-EOMCCSD(4h-2p)* routine
            if cvsmin != -1 and cvsmax != -1:
                R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n]], T, omega0[state_index[n]], fock, g, H1, H2, o, v, cvsmin, cvsmax, r3_excitations, maxit, convergence, max_size=max_size, out_of_core=out_of_core)
            else:
                R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n]], T, omega0[state_index[n]], fock, g, H1, H2, o, v, r3_excitations, maxit, convergence, max_size=max_size, out_of_core=out_of_core)
        else: # All other EOMCC calculations using conventional Davidson
            if r3_excitations is not None:
                if cvsmin != -1 and cvsmax != -1:
                    R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n]], T, omega0[state_index[n]], H1, H2, o, v, cvsmin, cvsmax, r3_excitations, maxit, convergence, max_size=max_size, out_of_core=out_of_core)
                else:
                    R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n]], T, omega0[state_index[n]], H1, H2, o, v, r3_excitations, maxit, convergence, max_size=max_size, out_of_core=out_of_core)
            else:
                if cvsmin != -1 and cvsmax != -1:
                    R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n]], T, omega0[state_index[n]], H1, H2, o, v, cvsmin, cvsmax, maxit, convergence, max_size=max_size, out_of_core=out_of_core)
                else:
                    R[n], omega[n], r0[n], rel = calculation(R0[:, state_index[n]], T, omega0[state_index[n]], H1, H2, o, v, maxit, convergence, max_size=max_size, out_of_core=out_of_core)
        toc = time.time()

        minutes, seconds = divmod(toc - tic, 60)

        print("")
        print("    EOMCC Excitation Energy: {: 20.12f}".format(omega[n]))
        print("    r0 = {: 20.12f}".format(r0[n]))
        print("    REL = {: 20.12f}".format(rel))
        print("")
        print("    Largest Singly and Doubly Excited Amplitudes")
        print("    --------------------------------------------")
        if method.lower() in ["eomccsd", "eomccsdt", "eomrccsd", "eomrccsdt", "eomcc3", "eomcc3-lin"]:
            print_amplitudes(R[n][0], R[n][1], PRINT_THRESH, rhf=flag_rhf)
        if method.lower() in ["dipeom3", "dipeom3-cvs", "dipeom4", "dipeom4_p", "dipeom4-cvs", "dipeom4_star_p"]:
            print_dip_amplitudes(R[n][0], R[n][1], PRINT_THRESH)
        print("")
        print("    EOMCC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
        print(f"    Memory usage: {get_memory_usage()} MB")
        print("")

    return R, omega, r0

def run_lefteomcc_calc(R, omega0, T, H1, H2, o, v, method, fock=None, g=None, maxit=80, convergence=1.0e-07, max_size=20, diis_size=6, do_diis=True, r3_excitations=None):
    from miniccpy.printing import print_amplitudes
    from miniccpy.utilities import biorthogonalize
    # check if requested EOMCC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    if method in RHF_MODULES:
        flag_rhf = True
    else:
        flag_rhf = False
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')

    nroot = len(R)

    L = [0 for i in range(nroot)]
    omega = [0 for i in range(nroot)]
    for n in range(nroot):
        print(f"    Solving for state #{n + 1}")
        tic = time.time()
        if method.lower()  == "left_eomcc3-lin": # Linear EOMCC3 model using conventional Davidson diagonalization
            L[n], omega[n] = calculation(R[n], T, omega0[n], fock, g, H1, H2, o, v, maxit, convergence, max_size=max_size)
        elif method.lower() == "left_eomcc3": # Folded EOMCC3 model using excited-state DIIS algorithm
            L[n], omega[n] = calculation(R[n], T, omega0[n], fock, g, H1, H2, o, v, maxit, convergence, diis_size=diis_size, do_diis=do_diis)
        else:
            L[n], omega[n] = calculation(R[n], T, omega0[n], H1, H2, o, v, maxit, convergence, max_size=max_size)
        toc = time.time()

        minutes, seconds = divmod(toc - tic, 60)

        print("")
        print("    Left-EOMCC Excitation Energy: {: 20.12f}".format(omega[n]))
        print("")
        print("    Largest Singly and Doubly Excited Amplitudes")
        print("    --------------------------------------------")
        if method.lower() in ["left_eomccsd", "left_eomccsdt", "left_eomrccsd", "left_eomrccsdt", "left_eomcc3", "left_eomcc3-lin"]:
            print_amplitudes(L[n][0], L[n][1], PRINT_THRESH, rhf=flag_rhf)
        print("")
        print("    Left-EOMCC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
        print(f"    Memory usage: {get_memory_usage()} MB")
        # check that the right eigenvalue is equal to the left eigenvalue
        print("    Check: |E(right) - E(left)| = ", abs(omega0[n] - omega[n]))
        assert np.allclose(omega0[n], omega[n], atol=1.0e-06)
        print("")

    #print("   Biorthonormality Check")
    #Rmat = np.asarray([np.hstack([r1.flatten(), r2.flatten()]) for r1, r2 in R]).T
    #Lmat = np.asarray([np.hstack([l1.flatten(), l2.flatten()]) for l1, l2 in L])
    #Lmat = biorthogonalize(Lmat, Rmat)
    #print("   |LR - 1| = ", np.linalg.norm(np.dot(Lmat, Rmat) - np.eye(nroot)))

    return L, omega

def run_correction(T, L, fock, H1, H2, o, v, method, **kwargs): 
    """Run the ground-state CC correction specified by `method`."""

    # check if requested CC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')

    tic = time.time()
    e_correction = calculation(T, L, fock, H1, H2, o, v, **kwargs)
    toc = time.time()
    minutes, seconds = divmod(toc - tic, 60)

    print("")
    for key, value in e_correction.items():
        print(f"    Triples correction energy ({key}): {value}")
    print("")
    print("    CC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
    print(f"    Memory usage: {get_memory_usage()} MB")
    print("")
    return e_correction

def run_eom_correction(T, R, L, r0, omega, fock, H1, H2, o, v, method, g=None):
    """Run the excited-state EOMCC correction specified by `method`."""

    # check if requested CC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')

    tic = time.time()
    if method == "eomccsdta_star":
        e_correction = calculation(T, R, L, r0, omega, fock, g, H1, H2, o, v)
    else:
        e_correction = calculation(T, R, L, r0, omega, fock, H1, H2, o, v)
    toc = time.time()
    minutes, seconds = divmod(toc - tic, 60)

    print("")
    for key, value in e_correction.items():
        print(f"    Triples correction energy ({key}): {value}")
    print("")
    print("    EOMCC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
    print(f"    Memory usage: {get_memory_usage()} MB")
    print("")
    return e_correction

def run_ip_correction(T, R, L, omega, fock, g, H1, H2, o, v, method):
    """Run the IP correction specified by `method`."""

    # check if requested CC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')

    tic = time.time()
    e_correction = calculation(T, R, L, omega, fock, g, H1, H2, o, v)
    toc = time.time()
    minutes, seconds = divmod(toc - tic, 60)

    print("")
    for key, value in e_correction.items():
        print(f"    3h-2p correction energy ({key}): {value}")
    print("")
    print("    IP-EOMCC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
    print(f"    Memory usage: {get_memory_usage()} MB")
    print("")
    return e_correction

def run_ea_correction(T, R, L, omega, fock, g, H1, H2, o, v, method):
    """Run the EA correction specified by `method`."""

    # check if requested CC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')

    tic = time.time()
    e_correction = calculation(T, R, L, omega, fock, g, H1, H2, o, v)
    toc = time.time()
    minutes, seconds = divmod(toc - tic, 60)

    print("")
    for key, value in e_correction.items():
        print(f"    3p-2h correction energy ({key}): {value}")
    print("")
    print("    EA-EOMCC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
    print(f"    Memory usage: {get_memory_usage()} MB")
    print("")
    return e_correction

def run_dip_correction(T, R, L, omega, fock, g, H1, H2, o, v, method):
    """Run the DIP correction specified by `method`."""

    # check if requested CC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')

    tic = time.time()
    e_correction = calculation(T, R, L, omega, fock, g, H1, H2, o, v)
    toc = time.time()
    minutes, seconds = divmod(toc - tic, 60)

    print("")
    for key, value in e_correction.items():
        print(f"    4h-2p correction energy ({key}): {value}")
    print("")
    print("    DIP-EOMCC calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
    print(f"    Memory usage: {get_memory_usage()} MB")
    print("")
    return e_correction

def run_lrcc1_calc(T, H1, H2, W, o, v, method, maxit=80, convergence=1.0e-07, energy_shift=0.0, diis_size=6, n_start_diis=0, out_of_core=False):
    """Run the ground-state LR-CC(1) calculation specified by `method`."""
    from miniccpy.printing import print_amplitudes

    # check if requested CC calculation is implemented in modules
    if method not in MODULES:
        raise NotImplementedError(
            "{} not implemented".format(method)
        )
    if method in RHF_MODULES:
        flag_rhf = True
    else:
        flag_rhf = False
    # import the specific CC method module and get its update function
    mod = import_module("miniccpy."+method.lower())
    calculation = getattr(mod, 'kernel')

    # Turn off DIIS for small systems; it becomes singular!
    if H1.shape[0] <= 4: 
        print("Turning off DIIS acceleration for small system")
        diis_size = 1000 

    tic = time.time()
    eta, prop_corr = calculation(T, H1, H2, W, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core)
    toc = time.time()

    minutes, seconds = divmod(toc - tic, 60)

    print("")
    print("    LR-CC(1) Correlation Energy Derivative: {: 20.12f}".format(prop_corr))
    print("")
    print("    Largest Singly and Doubly Excited Amplitudes")
    print("    --------------------------------------------")
    print_amplitudes(T[0], T[1], PRINT_THRESH, rhf=flag_rhf)
    print("")
    print("    LR-CC(1) calculation completed in {:.2f}m {:.2f}s".format(minutes, seconds))
    print(f"    Memory usage: {get_memory_usage()} MB")
    print("")

    return eta, prop_corr

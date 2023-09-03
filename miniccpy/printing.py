import numpy as np
import datetime
from pyscf import symm

WHITESPACE = "  "

def print_system_information(meanfield, nfrozen, hf_energy):
    """Print a nice output of the molecular system information."""

    molecule = meanfield.mol
    nelectrons = molecule.nelectron
    norbitals = meanfield.mo_coeff.shape[1]
    orbital_symmetries = symm.label_orb_symm(molecule, molecule.irrep_name, molecule.symm_orb, meanfield.mo_coeff)

    print(WHITESPACE, "System Information:")
    print(WHITESPACE, "----------------------------------------------------")
    print(WHITESPACE, "  Number of correlated electrons =", nelectrons - 2 * nfrozen)
    print(WHITESPACE, "  Number of correlated orbitals =", 2 * norbitals - 2 * nfrozen)
    print(WHITESPACE, "  Number of frozen orbitals =", 2 * nfrozen)
    print(
            WHITESPACE,
            "  Number of occupied orbitals =",
            nelectrons - 2 * nfrozen,
    )
    print(
            WHITESPACE,
            "  Number of unoccupied orbitals =",
            2 * norbitals - nelectrons ,
    )
    print(WHITESPACE, "  Charge =", molecule.charge)
    print(WHITESPACE, "  Symmetry group =", molecule.groupname.upper())
    print(
            WHITESPACE, "  Spin multiplicity of reference =", molecule.spin + 1
    )
    print("")

    HEADER_FMT = "{:>10} {:>20} {:>13} {:>13}"
    MO_FMT = "{:>10} {:>20.6f} {:>13} {:>13.1f}"

    header = HEADER_FMT.format("MO #", "Energy (a.u.)", "Symmetry", "Occupation")
    print(header)
    print(WHITESPACE + len(header) * "-")
    for i in range(norbitals):
        print(
                MO_FMT.format(
                    i + 1,
                    meanfield.mo_energy[i],
                    orbital_symmetries[i],
                    meanfield.mo_occ[i],
                )
        )
    print("")
    print(WHITESPACE, "Nuclear Repulsion Energy =", molecule.energy_nuc())
    print(WHITESPACE, "Reference Energy =", hf_energy)
    print("")

def print_custom_system_information(fock, nelectrons, nfrozen, hf_energy):
    """Print a nice output of the custom MO-defined molecular system."""

    norbitals = fock.shape[0]
    mo_energy = np.diag(fock)
    orbital_symmetries = ["C1"] * norbitals
    mo_occ = np.zeros(norbitals)
    mo_occ[:nelectrons] = 1.0

    print(WHITESPACE, "System Information:")
    print(WHITESPACE, "----------------------------------------------------")
    print(WHITESPACE, "  Number of correlated electrons =", nelectrons - 2 * nfrozen)
    print(WHITESPACE, "  Number of correlated orbitals =", norbitals - 2 * nfrozen)
    print(WHITESPACE, "  Number of frozen orbitals =", 2 * nfrozen)
    print(
            WHITESPACE,
            "  Number of occupied orbitals =",
            nelectrons - 2 * nfrozen,
    )
    print(
            WHITESPACE,
            "  Number of unoccupied orbitals =",
            norbitals - nelectrons ,
    )
    print("")

    HEADER_FMT = "{:>10} {:>20} {:>13} {:>13}"
    MO_FMT = "{:>10} {:>20.6f} {:>13} {:>13.1f}"

    header = HEADER_FMT.format("MO #", "Energy (a.u.)", "Symmetry", "Occupation")
    print(header)
    print(WHITESPACE + len(header) * "-")
    spin_label = ["𝜶", "𝜷"]
    for i in range(norbitals):
        spatial_i = (i // 2)
        print(
                MO_FMT.format(
                    str(i + 1) + "(" + str(spatial_i + 1) + spin_label[i % 2] + ")",
                    mo_energy[i],
                    orbital_symmetries[i],
                    mo_occ[i],
                )
        )
    print("")
    print(WHITESPACE, "Reference Energy =", hf_energy)
    print("")

def print_cis_vector(r, print_threshold):
    nu, no = r.shape
    n = 1
    for a in range(nu):
        for i in range(no):
            if abs(r[a, i]) > print_threshold:
                print(f"     [{n}]  {spatial_index(i + 1)}{spin_label(i + 1)} -> {spatial_index(a + no + 1)}{spin_label(a + no + 1)}    {r[a, i]}") 
                n += 1
    return 

def print_1p_vector(r, no, print_threshold):
    nu, = r.shape
    n = 1
    for a in range(nu):
        if abs(r[a]) > print_threshold:
            print(f"     [{n}]  -> {spatial_index(a + no + 1)}{spin_label(a + no + 1)}    {r[a]}") 
            n += 1
    return 

def print_1h_vector(r, nu, print_threshold):
    no, = r.shape
    n = 1
    for i in range(no):
        if abs(r[i]) > print_threshold:
            print(f"     [{n}]  {spatial_index(i + 1)}{spin_label(i + 1)} ->     {r[i]}") 
            n += 1
    return 

def spatial_index(p):
    if p % 2 == 0:
        return int(p / 2)
    else:
        return int((p + 1) / 2)

def spin_label(p):
    if p % 2 == 0:
        return "B"
    else:
        return "A"

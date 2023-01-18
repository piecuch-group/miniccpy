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
    print(WHITESPACE, "  Point group =", molecule.groupname.upper())
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


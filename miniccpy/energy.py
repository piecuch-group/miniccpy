import numpy as np

def cc_energy(t1, t2, f, g, o, v):
    """ Calculate the ground-state CC correlation energy defined by
    < 0 | exp(-T) H_N exp(T) | 0> = < i | f | a > < a | t1 | i > 
                            + 1/4 * < ij | v | ab > < ab | t2 | ij >
                            + 1/2 * < ij | v | ab > < a | t1 | i > < b | t1 | j >.
    """
    energy = np.einsum('ia,ai->', f[o, v], t1)
    energy += 0.25 * np.einsum('ijab,abij->', g[o, o, v, v], t2)
    energy += 0.5 * np.einsum('ijab,ai,bj->', g[o, o, v, v], t1, t1)

    return energy

def calc_r0(r1, r2, H1, H2, omega, o, v):
    """Calculate the zero-body component of the EOM excitation operator,
    r0 = 1/omega * < 0 | (H(CC) * R)_C | 0 >, where H(CC) = [H_N*exp(T)]_C
    and
    < 0 | (H(CC) * R)_C | 0 > = < i | H1 | a > < a | r1 | i >
                        + 1/4 * < ij | H2 | ab > < ab | r2 | ij >."""
    r0 = np.einsum("me,em->", H1[o, v], r1)
    r0 += 0.25 * np.einsum("mnef,efmn->", H2[o, o, v, v], r2)

    return r0/omega

def calc_rel(r0, r1, r2):
    """Calculate the relative excitation level (REL)"""
    rel_0 = r0**2
    rel_1 = np.einsum("ai,ai->", r1, r1, optimize=True)
    rel_2 = 0.25 * np.einsum("abij,abij->", r2, r2, optimize=True)
    rel = (rel_1 + 2.0 * rel_2)/(rel_0 + rel_1 + rel_2)
    return rel

def calc_rel_ea(r1, r2):
    """Calculate the relative excitation level (REL) for EA-EOMCC calculations"""
    rel_1 = np.einsum("a,a->", r1, r1, optimize=True)
    rel_2 = 0.5 * np.einsum("abj,abj->", r2, r2, optimize=True)
    rel = (rel_1 + 2.0 * rel_2)/(rel_1 + rel_2)
    return rel

def calc_rel_ip(r1, r2):
    """Calculate the relative excitation level (REL) for IP-EOMCC calculations"""
    rel_1 = -np.einsum("i,i->", r1, r1, optimize=True)
    rel_2 = -0.5 * np.einsum("ibj,ibj->", r2, r2, optimize=True)
    rel = (rel_1 + 2.0 * rel_2)/(rel_1 + rel_2)
    return rel

def hf_energy(z, g, o):
    """Calculate the Hartree-Fock energy using the molecular orbital
    integrals in un-normal order (i.e., using Z and V), defined as
    < 0 | H | 0 > = < i | z | i > + 1/2 * < ij | v | ij >."""
    energy = np.einsum('ii->', z[o, o])
    energy += 0.5 * np.einsum('ijij->', g[o, o, o, o])

    return energy

def hf_energy_from_fock(f, g, o):
    """Calculate the Hartree-Fock energy using the molecular orbital
    integrals in normal-order (i.e., using F and V), defined by
    < 0 | H | 0 > = < i | f | i > - 1/2 * < ij | v | ij >."""
    energy = np.einsum('ii->', f[o, o])
    energy -= 0.5 * np.einsum('ijij->', g[o, o, o, o])

    return energy

import numpy as np

def cc_energy_from_rdm(rdm1, rdm2, fock, g, o, v):
    # orbital slicing (correlated section, excluding frozen core)
    no, nu = fock[o, v].shape
    corr_o = slice(0, no)
    corr_v = slice(no, no + nu)

    # One-electron energy
    e_ov = np.einsum("ia,ia->", fock[o, v], rdm1[corr_o, corr_v])
    e_vo = np.einsum("ai,ai->", fock[v, o], rdm1[corr_v, corr_o])
    e_oo = np.einsum('ij,ij->', fock[o, o], rdm1[corr_o, corr_o])
    e_vv = np.einsum('ab,ab->', fock[v, v], rdm1[corr_v, corr_v])
    onebody = e_ov + e_vo + e_oo + e_vv
    # Two-electron energy
    e_oooo = 0.5 * np.einsum('ijkl,ijkl->', g[o, o, o, o], rdm2[corr_o, corr_o, corr_o, corr_o])
    e_vvvv = 0.5 * np.einsum('abcd,abcd->', g[v, v, v, v], rdm2[corr_v, corr_v, corr_v, corr_v])
    e_ooov = np.einsum('ijka,ijka->', g[o, o, o, v], rdm2[corr_o, corr_o, corr_o, corr_v])
    e_vvvo = np.einsum('abci,abci->', g[v, v, v, o], rdm2[corr_v, corr_v, corr_v, corr_o])
    e_ovov = np.einsum('iajb,iajb->', g[o, v, o, v], rdm2[corr_o, corr_v, corr_o, corr_v])
    e_oovv = 0.5 * np.einsum('ijab,ijab->', g[o, o, v, v], rdm2[corr_o, corr_o, corr_v, corr_v])
    twobody = e_oooo + e_vvvv + e_ooov + e_vvvo + e_ovov + e_oovv
    energy = onebody + twobody
    return energy

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

def lccsd_energy(l1, l2, lh1, lh2):
    energy = np.sqrt( np.sum(lh1.flatten()**2) + np.sum(lh2.flatten()**2) ) / np.sqrt( np.sum(l1.flatten()**2) + np.sum(l2.flatten()**2) )
    return energy

def ccd_energy(t2, g, o, v):
    """ Calculate the ground-state CC correlation energy defined by
    < 0 | exp(-T2) H_N exp(T2) | 0> = 1/4 * < ij | v | ab > < ab | t2 | ij >
    """
    energy = 0.25 * np.einsum('ijab,abij->', g[o, o, v, v], t2)

    return energy

def rcc_energy(t1, t2, f, g, o, v):
    """ Calculate the ground-state RHF-based CC correlation energy defined by
    < 0 | exp(-T) H_N exp(T) | 0> = < i | f | a > < a | t1 | i > 
                            + 1/4 * < ij | v | ab > < ab | t2 | ij >
                            + 1/2 * < ij | v | ab > < a | t1 | i > < b | t1 | j >.
    """
    v_ss = 2.0 * g[o, o, v, v] - np.transpose(g[o, o, v, v], (0, 1, 3, 2))
    tau = t2 + np.einsum("ai,bj->abij", t1, t1, optimize=True)
    energy = 2.0 * np.einsum("ia,ai->", f[o, v], t1, optimize=True)
    energy += np.einsum("ijab,abij->", v_ss, tau, optimize=True)
    return energy

def rccd_energy(t2, g, o, v):
    """ Calculate the ground-state RHF-based CC correlation energy defined by
    < 0 | exp(-T2) H_N exp(T2) | 0> = < ij | v | ab > < ab | t2 | ij >
    """
    v_ss = 2.0 * g[o, o, v, v] - np.transpose(g[o, o, v, v], (0, 1, 3, 2))
    energy = np.einsum("ijab,abij->", v_ss, t2, optimize=True)
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

def calc_r0_rhf(r1, r2, H1, H2, omega, o, v):
    """Calculate the zero-body component of the EOM excitation operator,
    r0 = 1/omega * < 0 | (H(CC) * R)_C | 0 >, where H(CC) = [H_N*exp(T)]_C"""
    v_ss = 2.0 * H2[o, o, v, v] - np.transpose(H2[o, o, v, v], (0, 1, 3, 2))
    r0 = 2.0 * np.einsum("me,em->", H1[o, v], r1)
    r0 += np.einsum("mnef,efmn->", v_ss, r2)
    return r0/omega

def calc_rel(r0, r1, r2):
    """Calculate the relative excitation level (REL)"""
    rel_0 = r0**2
    rel_1 = np.einsum("ai,ai->", r1, r1, optimize=True)
    rel_2 = 0.25 * np.einsum("abij,abij->", r2, r2, optimize=True)
    rel = (rel_1 + 2.0 * rel_2)/(rel_0 + rel_1 + rel_2)
    return rel

def calc_rel_rhf(r0, r1, r2):
    """Calculate the relative excitation level (REL) for RHF EOMCC."""
    rel_0 = r0**2
    rel_1 = 2.0 * np.einsum("ai,ai->", r1, r1, optimize=True)
    rel_2 = (
                2.0 * np.einsum("abij,abij->", r2, r2, optimize=True)
                    - np.einsum("abij,abji->", r2, r2, optimize=True)
    )
    rel = (rel_1 + 2.0 * rel_2)/(rel_0 + rel_1 + rel_2)
    return rel

def calc_rel_ea(r1, r2):
    """Calculate the relative excitation level (REL) for EA-EOMCC calculations"""
    rel_1 = np.einsum("a,a->", r1, r1, optimize=True)
    rel_2 = 0.5 * np.einsum("abj,abj->", r2, r2, optimize=True)
    rel = (rel_1 + 2.0 * rel_2)/(rel_1 + rel_2)
    return rel

def calc_rel_dea(r1, r2):
    """Calculate the relative excitation level (REL) for DEA-EOMCC calculations"""
    rel_1 = 0.5 * np.einsum("ab,ab->", r1, r1, optimize=True)
    rel_2 = (1.0 / 6.0) * np.einsum("abck,abck->", r2, r2, optimize=True)
    rel = (rel_1 + 2.0 * rel_2)/(rel_1 + rel_2)
    return rel

def calc_rel_ip(r1, r2):
    """Calculate the relative excitation level (REL) for IP-EOMCC calculations"""
    rel_1 = -np.einsum("i,i->", r1, r1, optimize=True)
    rel_2 = -0.5 * np.einsum("ibj,ibj->", r2, r2, optimize=True)
    rel = (rel_1 + 2.0 * rel_2)/(rel_1 + rel_2)
    return rel

def calc_rel_dip(r1, r2):
    """Calculate the relative excitation level (REL) for DIP-EOMCC calculations"""
    rel_1 = 0.5 * np.einsum("ij,ij->", r1, r1, optimize=True)
    rel_2 = (1.0 / 6.0) * np.einsum("ijck,ijck->", r2, r2, optimize=True)
    rel = (rel_1 + 2.0 * rel_2)/(rel_1 + rel_2)
    return rel

def hf_energy(z, g, o):
    """Calculate the Hartree-Fock energy using the molecular orbital
    integrals in un-normal order (i.e., using Z and V), defined as
    < 0 | H | 0 > = < i | z | i > + 1/2 * < ij | v | ij >."""
    energy = np.einsum('ii->', z[o, o])
    energy += 0.5 * np.einsum('ijij->', g[o, o, o, o])

    return energy

def rhf_energy(z, g, o):
    """Calculate the RHF Hartree-Fock energy using the molecular orbital
    integrals in un-normal order (i.e., using Z and V), defined as
    < 0 | H | 0 > = 2 < i | z | i > + 2 < ij | v | ij > - < ij | v | ji >."""
    energy = 2.0 * np.einsum('ii->', z[o, o])
    energy += 2.0 * np.einsum('ijij->', g[o, o, o, o])
    energy -= np.einsum('ijji->', g[o, o, o, o])

    return energy

def hf_energy_from_fock(f, g, o):
    """Calculate the Hartree-Fock energy using the molecular orbital
    integrals in normal-order (i.e., using F and V), defined by
    < 0 | H | 0 > = < i | f | i > - 1/2 * < ij | v | ij >."""
    energy = np.einsum('ii->', f[o, o])
    energy -= 0.5 * np.einsum('ijij->', g[o, o, o, o])

    return energy






def cc_corr_energy_from_rdm(rdm1, rdm2, fock, g, o, v):
    """Calculate the CC CORRELATION energy using the molecular orbital
    integrals in un-normal order by RDM, defined as
    < 0 | (1+Lambda) exp(-T) H_N exp(T) | 0> = 
    sum(p,q) < p | f | q > * gamma_N(qp) + sum(p,q,r,s) 1/4 * < pq | g | rs > * Gamma_N(rspq)."""

    # orbital slicing (correlated section, excluding frozen core)
    no, nu = fock[o, v].shape
    corr_o = slice(0, no)
    corr_v = slice(no, no + nu)

    # One-electron energy
    e_ov = np.einsum("ia,ai->", fock[o, v], rdm1[corr_v, corr_o])
    e_vo = np.einsum("ai,ia->", fock[v, o], rdm1[corr_o, corr_v])
    e_oo = np.einsum('ij,ji->', fock[o, o], rdm1[corr_o, corr_o])
    e_vv = np.einsum('ab,ba->', fock[v, v], rdm1[corr_v, corr_v])
    onebody = e_ov + e_vo + e_oo + e_vv
    
    # Two-electron energy
    e_oooo = 0.25 * np.einsum('ijkl,klij->', g[o, o, o, o], rdm2[corr_o, corr_o, corr_o, corr_o])
    e_ooov = 0.5 * np.einsum('ijka,kaij->', g[o, o, o, v], rdm2[corr_o, corr_v, corr_o, corr_o])
    e_oovv = 0.25 * np.einsum('ijab,abij->', g[o, o, v, v], rdm2[corr_v, corr_v, corr_o, corr_o])

    e_ovoo = 0.5 * np.einsum('iajk,jkia->', g[o, v, o, o], rdm2[corr_o, corr_o, corr_o, corr_v])
    e_ovov = np.einsum('iajb,jbia->', g[o, v, o, v], rdm2[corr_o, corr_v, corr_o, corr_v])
    e_ovvv = 0.5 * np.einsum('iabc,bcia->', g[o, v, v, v], rdm2[corr_v, corr_v, corr_o, corr_v])

    e_vvoo = 0.25 * np.einsum('abij,ijab->', g[v, v, o, o], rdm2[corr_o, corr_o, corr_v, corr_v])
    e_vvov = 0.5 * np.einsum('abic,icab->', g[v, v, o, v], rdm2[corr_o, corr_v, corr_v, corr_v])
    e_vvvv = 0.25 * np.einsum('abcd,cdab->', g[v, v, v, v], rdm2[corr_v, corr_v, corr_v, corr_v])
   
    twobody = e_oooo + e_ooov + e_oovv + e_ovoo + e_ovov + e_ovvv + e_vvoo + e_vvov + e_vvvv
    corr_energy = onebody + twobody



    return corr_energy


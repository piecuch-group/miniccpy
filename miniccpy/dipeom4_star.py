import time
import numpy as np
from miniccpy.utilities import get_memory_usage
from miniccpy.lib import dipeom4_star_p

def kernel(T, R, L, omega, fock, g, H1, H2, o, v):
    # Compute the noniterative correction to DIP(3h-1p) energies
    t1, t2 = T
    r1, r2 = R

    print("    ==> DIP-EOMCC(4h-2p)(P)* noniterative correction <==")
    delta_star = build_HR3_noniterative(r1, r2, t1, t2, fock, g, omega, H1, H2, o, v)
    print(f"   DIP-EOMCCSD(3h-1p) Vertical DIP Energy:     {omega}")
    print(f"   DIP-EOMCCSD(4h-2p)* Correction:     {delta_star}")
    print(f"   DIP-EOMCCSD(4h-2p)* Vertical DIP Energy:     {omega + delta_star}")
    delta_4h2p = {"A": delta_star, "B": 0.0, "C": 0.0, "D": 0.0}
    return delta_4h2p

def build_HR3_noniterative(r1, r2, t1, t2, fock, g, omega, H1, H2, o, v):
    """Compute the projection of HR on 4h-2p excitations
        X[i, j, c, d, k, l] = < ijklcd | [ HBar(CCSD) * (R1 + R2 + R3) ]_C | 0 >,
        approximated complete to 3rd-order in MBPT (assuming 2h is 0th order). The
        resulting terms include (H[2]*R1)_C + (H[1]*R2)_C + (F_N*R3)_C.
    """
    # Intermediates
    # This term factorizes the 4-body HBar formed by (V_N*T2^2)_C, which enters at 3rd-order.
    # This should be removed too
    I_vv = (
            0.5 * np.einsum("mnef,mn->ef", H2[o, o, v, v], r1, optimize=True)
    )

    # I(ijmk)
    I_oooo = (
          (3.0 / 6.0) * np.einsum("nmke,ijem->ijnk", H2[o, o, o, v], r2, optimize=True) # includes T1
        - (3.0 / 6.0) * np.einsum("mnik,mj->ijnk", H2[o, o, o, o], r1, optimize=True) # includes T1 and T2
    )
    # antisymmetrize A(ijk)
    I_oooo -= np.transpose(I_oooo, (0, 3, 2, 1)) # A(jk)
    I_oooo -= np.transpose(I_oooo, (1, 0, 2, 3)) + np.transpose(I_oooo, (3, 1, 2, 0)) # A(i/jk)
    # This (V_N*R3)_C term is removed
    #I_oooo = dipeom4_star_p.dipeom4_star_p.build_i_oooo(I_oooo, r3, r3_excitations, H2[o, o, v, v])

    # I(ijce)
    I_oovv = (
        (1.0 / 2.0) * np.einsum("cmfe,ijem->ijcf", H2[v, o, v, v], r2, optimize=True) # includes T1
        + np.einsum("bmje,mk->jkbe", H2[v, o, o, v], r1, optimize=True) # includes T1 and T2
        + 0.5 * np.einsum("nmie,njcm->ijce", H2[o, o, o, v], r2, optimize=True) # includes T1
        + 0.25 * np.einsum("ef,edil->lidf", I_vv, t2, optimize=True) # remove 4-body HBar term
    )
    # antisymmetrize A(ij)
    I_oovv -= np.transpose(I_oovv, (1, 0, 2, 3))
    # This (V_N*R3)_C term is removed
    #I_oovv = dipeom4_star_p.dipeom4_star_p.build_i_oovv(I_oovv, r3, r3_excitations, H2[o, o, v, v])

    delta_star = dipeom4_star_p.dipeom4_star_p.build_hr4_p_noniterative(
            t2, r2, omega,
            fock[o, o], fock[v, v],
            g[v, v, o, v], g[v, o, o, o], I_oooo, I_oovv,
    )
    return delta_star

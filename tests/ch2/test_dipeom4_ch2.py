import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

def spatial(x):
    if x % 2 == 0:
        return x // 2
    else:
        return (x - 1) // 2

def test_dipeom4_ch2():

    basis = 'sto-3g'
    nfrozen = 0

    geom = [["C", (0.0, 0.0, 0.0)],
            ["H", (0.0, 1.644403, -1.32213)],
            ["H", (0.0, -1.644403, -1.32213)]]

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, symmetry="C2V", unit="Bohr", cartesian=False, charge=-2)

    T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')
    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

    nroot = 4
    R, omega_guess = run_guess(H1, H2, o, v, nroot, method="dipcis")
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="dipeom4", state_index=[0, 3], out_of_core=True)

    #
    # Check the results
    #
    #expected_vee = [-0.4700687744, -0.4490361545]
    #for i, vee in enumerate(expected_vee):
    #    assert np.allclose(omega[i], vee, atol=1.0e-06)

    t1, t2 = T
    nu, no = t1.shape
    r3_abaa = np.zeros((no // 2, no // 2, nu // 2, nu // 2, no // 2, no // 2))
    r1, r2, r3 = R[1]
    for i in range(no):
        isp = spatial(i)
        for j in range(i + 1, no):
            jsp = spatial(j)
            for k in range(j + 1, no):
                ksp = spatial(k)
                for l in range(k + 1, no):
                    lsp = spatial(l)
                    for c in range(nu):
                        csp = spatial(c)
                        for d in range(c + 1, nu):
                            dsp = spatial(d)
                            # abaa
                            if i % 2 == 0 and j % 2 == 1 and k % 2 == 0 and l % 2 == 0 and c % 2 == 0 and d % 2 == 0:
                                #print(i, j, "->", isp, jsp)
                                if abs(r3[i, j, c, d, k, l]) > 1.0e-06:
                                    print(isp, jsp, ksp, lsp, csp, dsp, ":", r3[i, j, c, d, k, l])
                            if i % 2 == 1 and j % 2 == 0 and k % 2 == 0 and l % 2 == 0 and c % 2 == 0 and d % 2 == 0:
                                if abs(r3[i, j, c, d, k, l]) > 1.0e-06:
                                    print(isp, jsp, ksp, lsp, csp, dsp, ":", r3[i, j, c, d, k, l])
                            if i % 2 == 0 and j % 2 == 0 and k % 2 == 1 and l % 2 == 0 and c % 2 == 0 and d % 2 == 0:
                                if abs(r3[i, j, c, d, k, l]) > 1.0e-06:
                                    print(isp, jsp, ksp, lsp, csp, dsp, ":", r3[i, j, c, d, k, l])
                            if i % 2 == 0 and j % 2 == 0 and k % 2 == 0 and l % 2 == 1 and c % 2 == 0 and d % 2 == 0:
                                if abs(r3[i, j, c, d, k, l]) > 1.0e-06:
                                    print(isp, jsp, ksp, lsp, csp, dsp, ":", r3[i, j, c, d, k, l])

if __name__ == "__main__":
    test_dipeom4_ch2()

import numpy as np
import time

def active_hole(x, nocc, nact):
    if x < nocc - nact:
        return 0
    else:
        return 1

def active_particle(x, nact):
    if x < nact:
        return 1
    else:
        return 0

def get_active_triples_pspace(nacto, nactu, no, nu, num_active=1):

    def count_active_occ(occ):
        return sum([active_hole(i, no, nacto) for i in occ])

    def count_active_unocc(unocc):
        return sum([active_particle(a, nactu) for a in unocc])

    print(f"   Constructing triples list for CCSDt({'I' * num_active})-type P space")
    print("   ---------------------------------------------------")
    print("   Number of active occupied orbitals = ", nacto)
    print("   Number of active unoccupied orbitals = ", nactu)

    tic = time.perf_counter()
    t3_excitations = []
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                if count_active_occ([i, j, k]) < num_active: continue
                for a in range(nu):
                    for b in range(a + 1, nu):
                        for c in range(b + 1, nu):
                            if count_active_unocc([a, b, c]) >= num_active:
                                t3_excitations.append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
    # Convert the spin-integrated lists into Numpy arrays
    t3_excitations = np.asarray(t3_excitations, order="F")
    if len(t3_excitations.shape) < 2:
        t3_excitations = np.ones((1, 6))
    # Print the number of triples of a given spincase 
    print(f"   Active space contains {t3_excitations.shape[0]} triples")
    toc = time.perf_counter()
    minutes, seconds = divmod(toc - tic, 60)
    print(f"   Completed in {minutes:.1f}m {seconds:.1f}s\n")
    return t3_excitations

def get_active_4h2p_pspace(nacto, no, nu, num_active=2):

    def count_active_occ(occ):
        return sum([active_hole(i, no, nacto) for i in occ])

    print(f"   Constructing triples list for DIP-EOMCCSD(4h-2p)({'I' * num_active})-type P space")
    print("   ---------------------------------------------------")
    print("   Number of active occupied orbitals = ", nacto)

    tic = time.perf_counter()
    r3_excitations = []
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                for l in range(k + 1, no):
                    if count_active_occ([i, j, k, l]) < num_active: continue
                    for c in range(nu):
                        for d in range(c + 1, nu):
                            r3_excitations.append([c + 1, d + 1, i + 1, j + 1, k + 1, l + 1])
    # Convert the spin-integrated lists into Numpy arrays
    r3_excitations = np.asarray(r3_excitations, order="F")
    if len(r3_excitations.shape) < 2:
        r3_excitations = np.ones((1, 6))
    # Print the number of triples of a given spincase 
    print(f"   Active space contains {r3_excitations.shape[0]} triples")
    toc = time.perf_counter()
    minutes, seconds = divmod(toc - tic, 60)
    print(f"   Completed in {minutes:.1f}m {seconds:.1f}s\n")
    return r3_excitations

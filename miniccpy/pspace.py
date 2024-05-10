import numpy as np
import time
from miniccpy.utilities import get_memory_usage

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

def get_active_triples_pspace(no, nu, nacto=0, nactu=0, num_active=1, point_group="C1", orbsym=None, target_irrep="A"):
    from miniccpy.symmetry import get_pg_irreps, get_reference_symmetry

    def count_active_occ(occ):
        return sum([active_hole(i, no, nacto) for i in occ])

    def count_active_unocc(unocc):
        return sum([active_particle(a, nactu) for a in unocc])


    pg_irrep_to_number = get_pg_irreps(point_group)
    if orbsym is None:
        orbsym = ["A" for i in range(no + nu)]

    isym = [pg_irrep_to_number[p] for p in orbsym]
    reference_irrep = get_reference_symmetry(no, point_group, isym)
    sym_target = pg_irrep_to_number[target_irrep]
    sym_ref = pg_irrep_to_number[reference_irrep]

    print(f"   Constructing triples list for CCSDt({'I' * num_active})-type P space")
    print("   ---------------------------------------------------")
    print("   Total number of occupied orbitals = ", no)
    print("   Total number of unoccupied orbitals = ", nu)
    print("   Number of active occupied orbitals = ", nacto)
    print("   Number of active unoccupied orbitals = ", nactu)
    print("   Reference Irrep = ", reference_irrep)
    print(f"   Target Irrep = {target_irrep} ({point_group})")

    tic = time.perf_counter()
    t3_excitations = []
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                if count_active_occ([i, j, k]) < num_active: continue
                sym_occ = sym_ref ^ isym[i] ^ isym[j] ^ isym[k]
                for a in range(nu):
                    for b in range(a + 1, nu):
                        for c in range(b + 1, nu):
                            if count_active_unocc([a, b, c]) >= num_active:
                                sym_unocc = isym[a + no] ^ isym[b + no] ^ isym[c + no]
                                if sym_occ ^ sym_unocc != sym_target: continue
                                t3_excitations.append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
    # Convert the lists into Numpy arrays
    t3_excitations = np.asarray(t3_excitations, order="F")
    if len(t3_excitations.shape) < 2:
        t3_excitations = np.ones((1, 6))
    # Print the number of triples of a given spincase 
    print(f"   Active space contains {t3_excitations.shape[0]} triples")
    toc = time.perf_counter()
    minutes, seconds = divmod(toc - tic, 60)
    print(f"    Memory usage: {get_memory_usage()} MB")
    print(f"   Completed in {minutes:.1f}m {seconds:.1f}s\n")
    return t3_excitations

def get_active_4h2p_pspace(no, nu, nacto=0, num_active=2, point_group="C1", orbsym=None, target_irrep="A"):
    from miniccpy.symmetry import get_pg_irreps, get_reference_symmetry

    def count_active_occ(occ):
        return sum([active_hole(i, no, nacto) for i in occ])

    pg_irrep_to_number = get_pg_irreps(point_group)
    if orbsym is None:
        orbsym = ["A" for i in range(no + nu)]

    isym = [pg_irrep_to_number[p] for p in orbsym]
    reference_irrep = get_reference_symmetry(no, point_group, isym)
    sym_target = pg_irrep_to_number[target_irrep]
    sym_ref = pg_irrep_to_number[reference_irrep]

    print(f"   Constructing triples list for DIP-EOMCCSD(4h-2p)({'I' * num_active})-type P space")
    print("   ---------------------------------------------------")
    print("   Total number of occupied orbitals = ", no)
    print("   Number of active occupied orbitals = ", nacto)
    print("   Reference Irrep = ", reference_irrep)
    print(f"   Target Irrep = {target_irrep} ({point_group})")

    tic = time.perf_counter()
    r3_excitations = []
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                for l in range(k + 1, no):
                    if count_active_occ([i, j, k, l]) < num_active: continue
                    sym_occ = sym_ref ^ isym[i] ^ isym[j] ^ isym[k] ^ isym[l]
                    for c in range(nu):
                        for d in range(c + 1, nu):
                            sym_unocc = isym[c + no] ^ isym[d + no]
                            if sym_occ ^ sym_unocc != sym_target: continue
                            r3_excitations.append([c + 1, d + 1, i + 1, j + 1, k + 1, l + 1])
    # Convert the lists into Numpy arrays
    r3_excitations = np.asarray(r3_excitations, order="F")
    if len(r3_excitations.shape) < 2:
        r3_excitations = np.ones((1, 6))
    # Print the number of triples of a given spincase 
    print(f"   Active space contains {r3_excitations.shape[0]} 4p2h excitations")
    toc = time.perf_counter()
    minutes, seconds = divmod(toc - tic, 60)
    print(f"    Memory usage: {get_memory_usage()} MB")
    print(f"   Completed in {minutes:.1f}m {seconds:.1f}s\n")
    return r3_excitations

def get_active_4h2p_pspace_array(no, nu, nacto=0, num_active=2, point_group="C1", orbsym=None, target_irrep="A"):
    from miniccpy.symmetry import get_pg_irreps, get_reference_symmetry

    def count_active_occ(occ):
        return sum([active_hole(i, no, nacto) for i in occ])

    pg_irrep_to_number = get_pg_irreps(point_group)
    if orbsym is None:
        orbsym = ["A" for i in range(no + nu)]

    isym = [pg_irrep_to_number[p] for p in orbsym]
    reference_irrep = get_reference_symmetry(no, point_group, isym)
    sym_target = pg_irrep_to_number[target_irrep]
    sym_ref = pg_irrep_to_number[reference_irrep]

    print(f"   Constructing triples list for DIP-EOMCCSD(4h-2p)({'I' * num_active})-type P space")
    print("   ---------------------------------------------------")
    print("   Total number of occupied orbitals = ", no)
    print("   Number of active occupied orbitals = ", nacto)
    print("   Reference Irrep = ", reference_irrep)
    print(f"   Target Irrep = {target_irrep} ({point_group})")

    tic = time.perf_counter()
    pspace = np.zeros((nu, nu, no, no, no, no), dtype=np.int32)
    cnt = 0
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                for l in range(k + 1, no):
                    if count_active_occ([i, j, k, l]) < num_active: continue
                    sym_occ = sym_ref ^ isym[i] ^ isym[j] ^ isym[k] ^ isym[l]
                    for c in range(nu):
                        for d in range(c + 1, nu):
                            sym_unocc = isym[c + no] ^ isym[d + no]
                            if sym_occ ^ sym_unocc != sym_target: continue
                            pspace[c, d, i, j, k, l] = 1
                            cnt += 1
    # Print the number of triples of a given spincase 
    print(f"   Active space contains {cnt} 4p2h excitations")
    toc = time.perf_counter()
    minutes, seconds = divmod(toc - tic, 60)
    print(f"    Memory usage: {get_memory_usage()} MB")
    print(f"   Completed in {minutes:.1f}m {seconds:.1f}s\n")
    return pspace

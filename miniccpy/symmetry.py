
def get_reference_symmetry(no, point_group, isym):
    # Get the point group symmetry of the reference state by exploiting
    # homomorphism between Abelian point groups and binary vector spaces
    # sym( irrep1, irrep2 ) = xor( irrep1, irrep2 ), where irreps are
    # numbered in the convention (for D2H):
    # Ag = 0, B1g = 1, B2g = 2, B3g = 3, Au = 4, B1u = 5, B2u = 6, B3u = 7

    pg_irrep_to_number = get_pg_irreps(point_group)
    pg_number_to_irrep = {v: k for k, v in pg_irrep_to_number.items()}

    sym = 0
    for i in range(no):
        sym = sym ^ isym[i]
    sym_ref = pg_number_to_irrep[sym]
    return sym_ref

def get_pg_irreps(pg):
    """Obtain Abelian symmetry point group irreps and their numerical labels. Using the
    binary arithmetic trick to compute group multiplication products, all gerade (g) labels
     should appear before ungerade (u) labels."""
    pg = pg.upper()
    pg_irreps = {
        "C1":  {"A": 0},
        "C2" : {"A": 0, "B": 1},
        "CI" : {"AG": 0, "AU": 1},
        "CS":  {"A'": 0, "A\"": 1},
        "C2V": {"A1": 0, "A2": 1, "B1": 2, "B2": 3},
        "C2H": {"AG": 0, "BG": 1, "BU": 2, "AU": 3},
        "D2" : {"A" : 0, "B1" : 1, "B2" : 2, "B3" : 3},
        "D2H": {"AG": 0, "B1G": 1, "B2G": 2, "B3G": 3, "AU" : 4, "B1U": 5, "B2U": 6, "B3U": 7},
    }
    return pg_irreps[pg]


def sym_table(pg):
    """Obtain the group multiplication table of the Abelian symmetry point group."""
    sym_mult = {
        "C1": [[0]],
        "C2": [[0, 1],
               [1, 0]],
        "CS": [[0, 1],
               [1, 0]],
        "CI": [[0, 1],
               [1, 0]],
        "C2V": [[0, 1, 2, 3],
                [1, 0, 3, 2],
                [2, 3, 0, 1],
                [3, 2, 1, 0]],
        "C2H": [[0, 1, 2, 3],
                [1, 0, 3, 2],
                [2, 3, 0, 1],
                [3, 2, 1, 0]],
        "D2" : [[0, 1, 2, 3],
                [1, 0, 3, 2],
                [2, 3, 0, 1],
                [3, 2, 1, 0]],
        "D2H": [[0, 1, 2, 3, 4, 5, 6, 7],
                [1, 0, 3, 2, 5, 4, 7, 6],
                [2, 3, 0, 1, 6, 7, 4, 5],
                [3, 2, 1, 0, 7, 6, 5, 4],
                [4, 5, 6, 7, 0, 1, 2, 3],
                [5, 4, 7, 6, 1, 0, 3, 2],
                [6, 7, 4, 5, 2, 3, 0, 1],
                [7, 6, 5, 4, 3, 2, 1, 0]],
    }

    return sym_mult[pg]

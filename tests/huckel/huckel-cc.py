import argparse

import numpy as np
from huckel import linear_huckel_model, cyclic_huckel_model
from miniccpy.driver import run_cc_calc

def polyene_cc(args):

    # Obtain either the linear or cyclic Huckel Hamiltonian
    if args.geom == "linear":
        fock, g, o, v, h1, h2 = linear_huckel_model(args.n)
    elif args.geom == "cyclic":
        fock, g, o, v, h1, h2 = cyclic_huckel_model(args.n)
    
    # Run the CC calculation
    T, E_corr = run_cc_calc(fock, g, o, v, method=args.method, convergence=1.0e-012)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the ground-state CC calculation on a Huckel model system.")
    parser.add_argument("geom", type=str, help="Specify whether to use 'linear' or 'cyclic' Huckel polyene model.")
    parser.add_argument("n", type=int, help="Number of sites in Huckel model (e.g., n = 2 is ethylene, n = 4 is butadiene, etc.")
    parser.add_argument("method", type=str, help="Ground-state CC method you want to run, e.g., 'ccsd', 'ccsdt', or 'ccsdtq'.")
    args = parser.parse_args()


    polyene_cc(args)




# Miniccpy:

Miniccpy is a small, general spin-orbital coupled-cluster (CC) package written in Python that implements
the standard ground-state single-reference CC approximations, such as CCSD, CCSDT, and CCSDTQ,
and a subset of their equation-of-motion (EOM) extensions, including EOMCCSD and EOMCCSDT, for describing
excited electronic states. At the moment, this package simply serves as a freely available demonstration
for how to program high-level CC/EOMCC methodologies in Python using Numpy. However, due to its generic
spin-orbital nature, it can, in principle, be used to perform fully relativistic calculations for 
electronic, or even nuclear, structure.

Miniccpy only deals with the correlated CC computation and uses Pyscf under the hood to obtain the starting 
Hartree-Fock solution and the associated one- and two-body molecular orbital integrals.

# Contact
Karthik Gururangan - gururang@msu.edu

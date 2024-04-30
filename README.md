# Miniccpy:

Miniccpy is a small coupled-cluster (CC) package written mostly in Python (along with some parts in Fortran) that implements
several ground-state single-reference CC and equation-of-motion (EOM) CC methodologies in addition
to some of the particle nonconserving EOMCC approaches of the singly and doubly electron attaching (EA)
and ionizing (IP) varieties. Miniccpy serves as a freely available demonstration for how to program high-level CC/EOMCC 
methodologies in Python using Numpy, and it is a useful sandbox for exploring the development and implementation
of new approaches. Miniccpy only deals with the correlated CC computation and uses interfaces to PySCF or GAMESS under the hood
to obtain the starting Hartree-Fock solution and the associated one- and two-body molecular orbital integrals.

## Spinorbital Computational Options:
For simplicity, most of the routines in Miniccpy utilize the most general
spinorbital form the CC/EOMCC equations, which does not assume that either the S<sub>z</sub> or S<sup>2</sup>
quantum numbers are conserved. Although the costs of the resulting calculations are substantially higher than those characterizing
the commonly used spin-integrated form of the equations, which accomplishes a partial spin-adaptation by eliminating
components of the equations that have different S<sub>z</sub> values than the reference state, the spinorbital formulation 
produces the simplest and most general form of the CC/EOMCC equations that is applicable to a variety of many-body computations. 
This includes performing CC/EOMCC calculations using general spin-broken reference states (e.g., generalized Hartree-Fock, or GHF)
as well as two-component relativistic approaches and nuclear structure calculations within the M-scheme. 

### Ground-state CC methodologies
- CCD
- CCSD
- CCSD(T)
- CR-CC(2,3)
- CC3
- CCSDT
- CCSDTQ
### Excited-state EOMCC methodologies 
- EOMCCSD
- CR-EOMCC(2,3)
- EOM-CC3
- EOMCCSDT
- IP-EOMCCSD(2h-1p)
- IP-EOMCCSD(3h-2p)
- EA-EOMCCSD(2p-1h)
- EA-EOMCCSD(3p-2h)
- DIP-EOMCCSD(3h-1p)
- DIP-EOMCCSD(4h-2p)
- DEA-EOMCCSD(3p-1h)
- DEA-EOMCCSD(4p-2h)

## RHF-based Non-orthogonally Spin-Adapted Computational Options:
Although Miniccpy is primarily a spinorbital CC code, it also offers a limited selection of computational options
that exploit the RHF-based non-orthogonally spin-adapted formulation. This spin-free form of the CC/EOMCC equations
allows for highly efficient computations, although it is only applicable to singlet states.

### Ground-state CC methodologies
- CCD
- CCSD
- CC3
- CCSDT
### Excited-state EOMCC methodologies 
- EOMCCSD
- EOMCCSDT

</p>

## Development Team

Karthik Gururangan  
Doctoral student, Department of Chemistry, Michigan State University  
e-mail: gururang@msu.edu  

Professor Piotr Piecuch  
University Distinguished Professor and Michigan State University Foundation Professor, Department of Chemistry, Michigan State University  
Adjunct Professor, Department of Physics and Astronomy, Michigan State University

<p align="justify">

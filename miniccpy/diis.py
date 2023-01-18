import numpy as np
from miniccpy.utilities import remove_file

class DIIS:
    """Class for the DIIS accelerator engine.

    Attributes
    ----------
    ndim : int
        Size of vector that is to be extrapolated
    diis_size : int
        Number of diis vectors to use in extrapolation
    out_of_core : bool
        Boolean to indicate whether or not to use disk memory to store vectors and residuals
    vecfile : str (default="t.npy")
        File name of Numpy memory map holding the previous vectors
    residfile : str (default="dt.npy")
        File name of Numpy memory map holding the previous residuals
    """
    def __init__(self, ndim, diis_size, out_of_core, vecfile="t.npy", residfile="dt.npy"):

        self.diis_size = diis_size
        self.out_of_core = out_of_core
        self.ndim = ndim
        self.vecfile = vecfile
        self.residfile = residfile

        if self.out_of_core:
            self.T_list = np.memmap(
                self.vecfile, mode="w+", dtype=np.float64, shape=(self.ndim, self.diis_size)
            )
            self.T_residuum_list = np.memmap(
                self.residfile, mode="w+", dtype=np.float64, shape=(self.ndim, self.diis_size)
            )
            self.flush()
        else:
            self.T_list = np.zeros((self.ndim, diis_size))
            self.T_residuum_list = np.zeros((self.ndim, diis_size))

    def cleanup(self):
        if self.out_of_core:
            remove_file(self.vecfile)
            remove_file(self.residfile)
            
    def push(self, T_tuple, T_residuum_tuple, iteration):
        self.T_list[:, iteration % self.diis_size] = np.hstack([t.flatten() for t in T_tuple])
        self.T_residuum_list[:, iteration % self.diis_size] = np.hstack([dt.flatten() for dt in T_residuum_tuple])
        if self.out_of_core:
            self.flush()

    def flush(self):
        self.T_list.flush()
        self.T_residuum_list.flush()

    def extrapolate(self):

        B_dim = self.diis_size + 1
        B = -1.0 * np.ones((B_dim, B_dim))

        nhalf = int(self.ndim / 2)
        for i in range(self.diis_size):
            for j in range(i, self.diis_size):
                B[i, j] = np.dot(
                    self.T_residuum_list[:nhalf, i], self.T_residuum_list[:nhalf, j]
                )
                B[i, j] += np.dot(
                    self.T_residuum_list[nhalf:, i], self.T_residuum_list[nhalf:, j]
                )
                B[j, i] = B[i, j]
        B[-1, -1] = 0.0

        rhs = np.zeros(B_dim)
        rhs[-1] = -1.0

        # TODO: replace with numpy.linalg.solve
        # TODO: replace with scipy.linalg.lu
        coeff = solve_gauss(B, rhs)
        x_xtrap = np.zeros(self.ndim)
        for i in range(self.diis_size):
            x_xtrap += coeff[i] * self.T_list[:, i]

        return x_xtrap

def solve_gauss(A, b):
    """DIIS helper function. Solves the linear system Ax=b using
    Gaussian elimination"""
    n = A.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, :] -= m * A[i, :]
            b[j] -= m * b[i]
    x = np.zeros(n)
    k = n - 1
    x[k] = b[k] / A[k, k]
    while k >= 0:
        x[k] = (b[k] - np.dot(A[k, k + 1 :], x[k + 1 :])) / A[k, k]
        k = k - 1

    return x

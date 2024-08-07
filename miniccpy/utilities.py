import numpy as np
import scipy

# gives a single float value
# psutil.cpu_percent()
# gives an object with many fields
# psutil.virtual_memory()
# you can convert that object to a dictionary
# dict(psutil.virtual_memory()._asdict())
# you can have the percentage of used RAM
# psutil.virtual_memory().percent
# 79.2
# you can calculate percentage of available memory
# psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
# 20.8

def biorthogonalize(L, R):
    LR = np.dot(L, R)
    # Left-modification scheme
    ML, MU = scipy.linalg.lu(LR, permute_l=True)
    return np.linalg.solve(MU, np.linalg.solve(ML, L))
    # Symmetric scheme
    # L = np.linalg.solve(ML, L)   # ML\L'
    # R = np.linalg.solve(MU, R.T) # R/MU


def get_memory_usage():
    """Returns the amount of memory currently used in MB. Useful for
    investigating the memory usages of various routines."""
    import os
    import psutil
    current_process = psutil.Process(os.getpid())
    memory = current_process.memory_info().rss
    return memory / (1024 * 1024)


def clean_up(fid, n):
    for i in range(n):
        remove_files(fid + "-" + str(i + 1) + ".npy")
    return


def remove_file(filePath):
    import os
    try:
        os.remove(filePath)
    except OSError:
        pass
    return

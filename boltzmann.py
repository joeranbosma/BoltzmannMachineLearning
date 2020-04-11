"""
Author: Joeran Bosma
"""

import numpy as np
import os
import importlib.util


# use appropriate sampler easily
def calculate_statistics(w, h, N, num_samples, cpp_sampler, num_threads=3, verbose=False):
    if cpp_sampler:
        # load C++ sampler
        spec = importlib.util.spec_from_file_location("Worker", os.environ['COMPILED_WORKER_PATH'])
        cpp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cpp)

        # create worker object with relevant couplings, external field and number of neurons
        worker = cpp.Worker(w, h, N, num_samples, num_threads)  # iterations, threads
        # calculate statistics
        stats = worker.get_stats()
        # split result
        m_est, S2_est = stats.getS(), stats.getSS()
    else:
        # calculate statistics using Python sampler
        m_est, S2_est, acceptance_ratio = MHMC_stats(w, h, iterations=num_samples, flip=1, verbose=verbose)

    return m_est, S2_est


# split sampling in multiple parts to reduce RAM footprint
def calculate_statistics_ram_friendlier(w, h, N, num_samples, cpp_sampler, num_threads, parts=10):
    # calculate statistics in parts
    m_est_list, S2_est_list = [], []
    for prt in range(parts):
        m_est, S2_est = calculate_statistics(w, h, N, num_samples=int(num_samples / parts),
                                             cpp_sampler=cpp_sampler, num_threads=num_threads)
        m_est_list.append(m_est)
        S2_est_list.append(S2_est)

    # return average of estimates
    return np.mean(np.array(m_est_list), axis=0), np.mean(np.array(S2_est_list), axis=0)


# convert number to binary rep, e.g. 13 ==> [1, 0, 1, 1, 0, ..] ==> [1, -1, 1, 1, -1, ..]
# binary rep is in reversed order: 1, 2, 4, 8, 16, ..
def decode(ids, N=11):
    ids = np.array(ids)

    to_scalar = False
    # catch single-element inputs
    if len(ids.shape) == 0:
        ids = np.array([ids])
        to_scalar = True

    M = np.zeros((len(ids), N), dtype=int)
    for k in range(N):
        M[:, k] = ids // 2 ** k % 2

    # convert 0 to -1, e.g. [1, 0, 1, 1, 0, ..] ==> [1, -1, 1, 1, -1, ..]
    M[M == 0] = -1

    if to_scalar:
        return M[0]
    else:
        return M


# convert binary rep to number, e.g. [1, 0, 1, 1, 0, ..] ==> 13
# both methods also work for lists of numbers/binary reps
def encode(M, N=11):
    M = np.array(M)

    to_scalar = False
    # handle both arrays of microstates, and single microstates
    if len(M.shape) == 1:
        M = np.array([M])
        to_scalar = True

    # convert -1 to 1, e.g. [1, -1, 1, 1, -1, ..] ==> [1, 0, 1, 1, 0, ..]
    M[M == -1] = 0

    ids = np.sum([M[:, k] * 2 ** k for k in range(N)], axis=0)
    if to_scalar:
        return ids[0]
    else:
        return ids

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:51:56 2022

@author: camer
"""

from itertools import product
import numpy as np
        
def factorize_matrix(matrix, k=2, steps=5000, alpha=0.0001, beta=0.001):
    '''Given a test matrix R with dimensions MxN, and the initial decomposed
    prediction matrix P dim. MxK and Q dim. KxN, compares P.Q to R and adjusts 
    using the factors alpha and beta over the given number of steps to converge
    error between the non-zero elements of R and P.Q to zero. Zero elements 
    are replaced with predictions.'''

    mse = []    
    matrix = np.array(matrix)
    m, n = matrix.shape
    P = np.random.rand(m, k)
    Q = np.random.rand(k, n)

    for _ in range(steps):
        # iterate over the rows of R s number of times
        for i, j in product(range(m), range(n)):   
            # if the current element is non-zero, see similarity to the training matrix
            if matrix[i, j] > 0:
                # get error between R and P.Q
                e = matrix[i, j] - P[i,:]@Q[:,j]
                # adjust P and Q and predict unrated items
                for k in range(k):
                    P[i, k] += alpha*(2*e*Q[k, j] - beta*P[i, k])
                    Q[k, j] += alpha*(2*e*P[i, k] - beta*Q[k, j])
        step_mse = np.mean((matrix - np.dot(P, Q))**2)
        mse.append(step_mse)
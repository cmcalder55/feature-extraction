
import numpy as np
from numpy.linalg import eig


_fmt_sample = lambda v: [np.array([i]).transpose() for i in v]

def format_sample_vector(data_vectors, flatten=True):
    if flatten:
        v_flat = sum(data_vectors, [])
        return _fmt_sample(v_flat)
    return [_fmt_sample(v) for v in data_vectors]

def mean_adjust_sample(samples):
    d = format_sample_vector(samples)             # reshape input data
    m = sum(d)/len(d)                  
    xk_m = np.subtract(d, m)
    return xk_m, m

def calc_scatter_matrix(xk_m):
    # sum dot prods of xk-m and its transpose 
    S_k = [i @ i.T for i in xk_m]       
    # sum column-wise to get the scatter matrix (S)       
    return np.sum(S_k, 0)

def create_pca_classifier(samples: list[list[int]]):
    # calculate mean-adjusted data vectors                
    xk_m, m = mean_adjust_sample(samples)

    # calculate scatter matrix
    S_w = calc_scatter_matrix(xk_m)

    # perform eigen decomp on S to get eigen values/vectors
    e_val, e_vec = eig(S_w)  
    e_top = np.amax(e_vec, 0)           # largest column

    # use max/dominant eigenvector to create the classifier 
    ak = e_top.T @ xk_m
    pca_classifier = m.T + ak*e_top
    return pca_classifier

def create_lda_classifier(samples):
    data = format_sample_vector(samples, flatten=False)
    v_mean = [sum(d)/len(d) for d in data]
    xk_m = [i-m for d,m in zip(data,v_mean) for i in d]
    S_w = calc_scatter_matrix(xk_m)

    # transform vector
    m_diff = v_mean[0] - v_mean[1:]
    w = np.linalg.inv(S_w) @ m_diff

    # sample projections
    yk = []                 
    for d in data:
        yk.extend(w[0].T @ d)
        
    lda_classifier = np.concatenate([y @ w[0].T for y in yk], 0)
    return lda_classifier

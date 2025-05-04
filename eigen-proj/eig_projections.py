
import numpy as np
from numpy.linalg import eig

def _prep_vector(d):
    return [np.array([i]).transpose() for i in d]

def projection_vector(data_vectors, d_sum):
    
    S_w = None
    m_diff = None
    for d in data_vectors:
        # mean vector; track mean difference        
        vm = sum(d)/len(d)
        if m_diff is None:
            m_diff = vm
        else:
            m_diff -= vm
        
        # difference between x_k and mean vector            
        xk = [i-vm for i in d]
        # weighted vector
        S_k = np.sum([i.dot(i.T) for i in xk], 0)
        if S_w is None:
            S_w = S_k
        else:
            S_w += S_k

    # calculate eigenvalue/vectors    
    Sb = np.dot(m_diff, m_diff.T)    
    e_val, e_vec = eig(np.dot(S_w.T, Sb))
    print(e_val)

    # dot prod of original data vectors and weighted
    w = np.amax(e_vec, 0)    
    return d_sum.dot(w)

def classify(d1, d2):
    
    d_sum = np.array(d1+d2)    
    data_vectors = [_prep_vector(d) for d in [d1, d2]]
    
    projection = projection_vector(data_vectors, d_sum)
    n = len(projection)

    class2_m = np.mean(projection)
    c2_proj = [((i - class2_m)**2)/n for i in projection]
    class2_sd = np.sum(c2_proj)

    den = np.sum([np.abs(x-1) for x in projection])
    class1_m = -n/(2*den)
    c1_proj = [np.abs((i - class1_m)**2-1) for i in projection]
    class1_d = -n/2*(np.sum(c1_proj))
    
    return ((class1_m, class1_d), (class2_m, class2_sd))

if __name__ == "__main__":
    
    # training samples D = {D1,D2}
    # d1 = [[12,7,3],[8,10,7],[10,11,9],[7,12,13],[11,9,10]]
    # d2 = [[1,4,5],[4,6,6],[1,7,5],[2,8,7],[3,2,5]]
    
    d1 = [[1,2], [-3,-1], [4,5], [-1,1]]
    d2 = [[0,-2], [5,2], [-1,-4], [3,1]]

    weighted_classes = classify(d1, d2)
    for c in weighted_classes:
        print(c)

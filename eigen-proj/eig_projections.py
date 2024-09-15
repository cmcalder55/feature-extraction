
import numpy as np
from numpy.linalg import eig

# training samples D = {D1,D2}
d1 = [[12,7,3],[8,10,7],[10,11,9],[7,12,13],[11,9,10]]
d2 = [[1,4,5],[4,6,6],[1,7,5],[2,8,7],[3,2,5]]

d_original = np.array(d1+d2)

d = lambda x: [np.array([i]).transpose() for i in x]

d1 = d(d1)
d2 = d(d2)

# mean vector
m1, m2 = sum(d1)/len(d1), sum(d2)/len(d2)
# difference between x_k and mean vector
xk_m1 = [i-m1 for i in d1]
xk_m2 = [i-m2 for i in d2]

Sk = lambda x: np.sum([i.dot(i.T) for i in x],0)
Sw = Sk(xk_m1) + Sk(xk_m2)

md = m1 - m2
Sb = np.dot(md,md.T)

e_val, e_vec = eig(np.dot(Sw.T,Sb))
w = np.amax(e_vec,0)

projection = d_original.dot(w)
n = len(projection)

class2_m = np.mean(projection)
class2_sd = np.sum([((i - class2_m)**2)/n for i in projection])

den = np.sum([np.abs(x-1) for x in projection])
class1_m = -n/(2*den)
class1_d = -n/2*(np.sum([np.abs((i - class1_m)**2-1) for i in projection]))

print((class1_m, class1_d), (class2_m, class2_sd))

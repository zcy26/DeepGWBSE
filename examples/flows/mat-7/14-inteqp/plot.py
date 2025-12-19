
import numpy as np
data = np.loadtxt('bandstructure.dat')
bands = data[:,1]
kpts = data[:,2:5]
emf = data[:,5]
eqp = data[:,6]
emf -= np.amax(emf[bands==4])
eqp -= np.amax(eqp[bands==4])
def get_x(ks):
    global dk_len
    dk_vec = np.diff(ks, axis=0)
    dk_len = np.linalg.norm(dk_vec, axis=1)
    return np.insert(np.cumsum(dk_len), 0, 0.)
xmin, xmax = np.inf, -np.inf
bands_uniq = np.unique(bands).astype(int)
f = open("band.dat","w")
for ib in bands_uniq:
    cond = bands==ib
    x = get_x(kpts[cond])
    xmin, xmax = min(xmin, x[0]), max(xmax, x[-1])
    for i_n in range(len(x)):
        f.write("%.9f %.9f %.9f \n" % (x[i_n], emf[cond][i_n], eqp[cond][i_n]))
    f.write("\n")
f.close()
    
from .dvr_vqe import DVR_VQE
import pennylane as qml
from .pot_gen_1d import *

def dvr_0inf(r, v, m=1, start_ind=0):
    import numpy as np
    dr = r[1] - r[0]
    N = r.shape[0]
    dvr = np.zeros((r.shape[0], r.shape[0]), dtype=float)
    f = 1 / (2 * m * np.square(dr))
    for i in range(N):
        for j in range(i):
            dvr[i, j] = (-1)**(i - j) * f * (2 / np.square(i - j) - 2 / (np.square(i + j + 2 * start_ind + 2)))
            dvr[j, i] = dvr[i, j]
        dvr[i, i] = v[i] + f * (np.square(np.pi) / 3 - 0.5 / (i + start_ind + 1)**2)
            
    return dvr

def get_dvr_r(dvr_options):
    import numpy as np
    box_max = dvr_options['box_lims'][1]
    box_min = dvr_options['box_lims'][0]
    count = int(box_max / dvr_options['dx'])

    r = np.linspace(0, box_max, count + 1)
    r_box = r[r > box_min]
    return r_box[:dvr_options['count']]

def get_ham_DVR(pot1d, dvr_options, mol_params):
    import numpy as np
    r_box = get_dvr_r(dvr_options)
    N = r_box.shape[0]
    start_ind = int(r_box[0] / dvr_options['dx']) - 1
    v = pot1d(r_box)
    return dvr_0inf(r_box, v, m=mol_params['mu'], start_ind=start_ind)

    

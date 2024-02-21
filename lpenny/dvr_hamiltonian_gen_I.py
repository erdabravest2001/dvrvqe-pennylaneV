from .dvr_vqe import DVR_VQE
import pennylane as qml
from pennylane import numpy as np
from .pot_gen_1d import *

def dvr_0inf(r, v, m=1, start_ind=0):
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
    box_max = dvr_options['box_lims'][1]
    box_min = dvr_options['box_lims'][0]
    count = int(box_max / dvr_options['dx'])

    r = np.linspace(0, box_max, count + 1)
    r_box = r[r > box_min]
    return r_box[:dvr_options['count']]

def get_ham_DVR(pot1d, dvr_options, mol_params):
    r_box = get_dvr_r(dvr_options)
    N = r_box.shape[0]
    start_ind = int(r_box[0] / dvr_options['dx']) - 1
    v = pot1d(r_box)
    return dvr_0inf(r_box, v, m=mol_params['mu'], start_ind=start_ind)

def gen_ham(mol_params, spin, dvr_options):
        mol_params['name'] += f'_{spin}'
        # obtain the potential for a CR2 at certain spin
        pot, lims = get_pot_cr2(spin)
        # perform a dvr vqe to obtain the hamiltonian
        dvr_vqe = DVR_VQE(mol_params, pot)
        h_dvr = dvr_vqe.get_h_dvr(dvr_options, J=0)*hartree
        h_dvr_p = qml.pauli_decompose(h_dvr)
        return h_dvr, h_dvr_p

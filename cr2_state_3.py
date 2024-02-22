# Import necessary libraries
import pennylane as qml
from pennylane import numpy as np
# import necessary libraries
from lpenny.dvr_hamiltonian_gen_I import * 
from lpenny.dvr_hamiltonian_gen_II import *
from lpenny.dvr_vqe import *
from lpenny.greedy_search_algorithm import * 
from lpenny.pot_gen_1d import *
from lpenny.pot_gen_2d import *

mol_params = cr2_params
params2 = [3.2, 4.5]
N = 5
spin2 = 3

dvr_options2 = { 'type': '1d',
        'box_lims': (params2[0], params2[1]),
        'dx': (params2[1] - params2[0]) / (2**N),
        'count':(2**N)
        } 

h_dvr2 =  gen_ham(mol_params, spin2, dvr_options2)
h_dvr_p2 = qml.pauli_decompose(h_dvr2)
print(h_dvr_p2)

ansatz_options_list2 = {
    'type': 'greedy',
    'constructive': True,
    'layers': 1,
    'num_keep': 1,
    'num_qubits': 5,
    'add_h': False,
    'add_sx': True,
    'add_rs': True,
    'samples': 10, 
    'reps': 1,
    'max_gate': 15
    }

vqe_options_list2 = {
    'max_iterations': 10000,
    'num_qubits': 5,
    'opt': optax.amsgrad(0.01) 
}

g2 = GreedyCircs(ansatz_options_list2, h_dvr_p2, log_dir='')
best_energies2, best_circs2, best_overall_circuit2, best_overall_energy2, best_overall_params2, best_overall_converge_cnts2, best_overall_converge_vals2, best_ansatz_string2 = g2.multiple_vqes(ansatz_options_list2, vqe_options_list2)


import matplotlib.pyplot as plt
import seaborn as sns
y = -9862.07
sns.set_theme(style="whitegrid", palette="husl")
sns.lineplot(y=best_overall_converge_vals2, x=best_overall_converge_cnts2, label='Greedy Search VQE Energy')
plt.axhline(y=y, color='r', linestyle='--', label=fr'Target Energy 1 ({y} cm$^{-1}$)')
plt.xlabel('Iteration (# of steps)')
plt.legend()
plt.ylabel(r'Energy (cm$^{-1}$)')
plt.title('VQE Energy Convergence for Cr$_2$')
plt.show()
plt.savefig('/Users/ethanelliotrajkumar/Documents/GitHub/dvrvqe-pennylaneV/plot.png')

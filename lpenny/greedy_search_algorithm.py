import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.config import config
config.update("jax_enable_x64", True)


class GreedyCircs:
    """
    Class representing a greedy search algorithm for building quantum circuits.
    """

    def __init__(self, ansatz_options, h_dvr_pd, log_dir=None):
        self.ansatz_options = ansatz_options
        self.h_dvr_pd = h_dvr_pd
        self.log_dir= log_dir # Ensure this is a valid PennyLane observable
    
    def gen_id(self):
        import time
        return str(int(time.time()))

    def build_circuit(self, c, h_dvr_pd, ansatz_options):
        num_qubits = c.shape[0]  # Number of qubits in the circuit
        depth = c.shape[1]     # Depth of the circuit
        n_params =  np.count_nonzero(c == 1) + num_qubits
        params = np.zeros(n_params, requires_grad=True)
        
        def circuit(params): 
            # Initialization based on ansatz options
            p = 0 
            if ansatz_options['add_h']:
                for i in range(num_qubits):
                    qml.Hadamard(wires=i)
            if ansatz_options['add_sx']:
                for i in range(num_qubits):
                    qml.SX(wires=i)
            if ansatz_options['add_rs']:
                for i in range(num_qubits):
                    qml.RZ(params[p], wires=i)
                    p += 1
            for i in range(depth):
                layer = c[:, i]
                for q, gate in enumerate(layer):
                    if gate == 1:
                        qml.RZ(params[p], wires=q)
                        p += 1
                    elif gate == 2:
                        qml.ECR(wires=[(q-1) % num_qubits, q])
                    elif gate == 3:
                        qml.SX(wires=q)
        return circuit, params

    def get_greedy_ansatz(self, ansatz_options, h_dvr_pd, best_circs, log_file=None):
        import numpy as np
        layer_list = [np.random.randint(0, ansatz_options['max_gate'] + 1, (ansatz_options['num_qubits'], 1)) 
                        for _ in range(ansatz_options['samples'])]
        circ_list = []
        if best_circs is None or len(best_circs) == 0:
            circ_list = layer_list
        else:
            for c in best_circs:
                circ_list.extend([np.concatenate([c, l], axis=-1) for l in layer_list])
        
        ansatz_list = []
        circ_list_final = []
        ansatz_string_list = []
        params_list = []    
        
        for index, c in enumerate(circ_list):
            ansatz, params = self.build_circuit(c, h_dvr_pd, ansatz_options)
            print("Circuit number: ", index)
            s = qml.draw(ansatz)(params)
            k = print(s) 
            ansatz_list.append(ansatz)
            circ_list_final.append(c)
            ansatz_string_list.append(s)
            params_list.append(params)   
        return circ_list_final, ansatz_list, params_list, ansatz_string_list


    def opt_vqe(self, c, h_dvr_pd, vqe_options, opt_params, log_file=None):
        dev = qml.device('lightning.qubit', wires=vqe_options['num_qubits'])

        @qml.qnode(dev)
        def cost_fn(params):
            c(params)
            return qml.expval(h_dvr_pd)

        # Ensure opt_params is a JAX array
        opt_params = opt_params
        values = [cost_fn(opt_params)]
        max_iterations = vqe_options['max_iterations']
        opt = vqe_options['opt']
        opt_state = opt.init(opt_params)

        counts = []

        for n in range(max_iterations):
            gradient = jax.grad(cost_fn)(opt_params)
            updates, opt_state = opt.update(gradient, opt_state)
            opt_params = optax.apply_updates(opt_params, updates)
            counts.append(n)
            values.append(cost_fn(opt_params))
            if n % 2 == 0:
                print(f"Step = {n},  Energy = {values[-1]:.8f} cm^{-1}")

        best_energy = min(values)
        return counts, values, opt_params, best_energy
    
    def greedy_vqe(self, ansatz_options, h_dvr_pd, vqe_options, log_file=None, prev_circs=None):
        circ_list, ansatz_list, params_list, ansatz_string_list = self.get_greedy_ansatz(ansatz_options, h_dvr_pd, prev_circs, log_file)
        best_params_list = []
        best_energies_list = []
        converge_cnts_list = []
        converge_vals_list = []
        best_ansatz_string_list = []
        for i, (a_circ, s) in enumerate(zip(ansatz_list, ansatz_string_list)):
            print(f"Optimizing Circuit {i}")
            print(s)
            initial_params = params_list[i]
            # Initialize opt_params if needed, for example with zeros or a specific strategy
            converge_cnts, converge_vals, optimized_params, best_energy = self.opt_vqe(c=a_circ, 
                                                                                        h_dvr_pd=h_dvr_pd,
                                                                                        vqe_options=vqe_options, 
                                                                                        opt_params=initial_params,
                                                                                        log_file=log_file)
            print(f"Optimized Circuit {i} Energy = {best_energy:.8f} cm^-1")
            best_params_list.append(optimized_params)
            best_energies_list.append(best_energy)
            converge_cnts_list.append(converge_cnts)
            converge_vals_list.append(converge_vals)
            best_ansatz_string_list.append(s)
            
            if log_file:
                with open(log_file, 'a') as file:
                    file.write(f"Circuit {i+1}: Best Energy = {best_energy:.8f}, Counts = {len(converge_cnts)}\n")
            else:
                print(f"Circuit {i+1}: Best Energy = {best_energy:.8f}, Counts = {len(converge_cnts)}")
        
        inds = np.argsort(best_energies_list)
        keep_inds = inds[:ansatz_options['num_keep']]
        best_circs = []  
        best_params = []
        best_energies = []      
        for ind in keep_inds:
            best_circs.append(circ_list[ind])
            converge_cnts.append(converge_cnts_list[ind])
            converge_vals.append(converge_vals_list[ind])
            best_params.append(best_params_list[ind])
            best_energies.append(best_energies_list[ind])
            best_ansatz_string_list.append(ansatz_string_list[ind])
        return best_circs, best_params, best_energies, converge_cnts, converge_vals_list, best_ansatz_string_list

    def multiple_vqes(self, ansatz_options, vqe_options):
        import os
        """
        Perform a greedy search optimization of quantum circuits over multiple layers.
        
        Parameters:
        - ansatz_options: Dictionary containing options for the ansatz circuits.
        - vqe_options: Dictionary containing options for the VQE optimization.
        - log_file: Optional; file path for logging the progress and results.
        """
        vqe_id = self.gen_id()
        cont=0
        
        if self.log_dir != None:
            log_dir_id = self.log_dir + 'greedy' + '/'
            offset = 0
            madedir = False
            while (not madedir) and (cont == 0):
                if not os.path.exists(log_dir_id):
                    os.mkdir(log_dir_id)
                    madedir = True
                else:
                    offset += 1
                    log_dir_id = self.log_dir + vqe_id + f'({offset})/'
            log_file = log_dir_id + 'vqe.txt'
        elif self.log_dir == None:
             log_file = None
        
        best_circs = None  # Initialize with no circuits
        best_circs = []
        best_energies = []
        best_params = []
        best_converge_cnts = []
        best_converge_vals = []
        best_ansatz_str = []
        for l in range(ansatz_options['layers']):
            if log_file:
                with open(log_file, 'a') as file:
                    file.write(f'Layer {l+1} ...\n')
            print(f'Layer {l+1} ...')
            # Use the greedy VQE optimization for the current layer
            best_circs, best_params, energies, converge_cnts, converge_vals, ansatz_str = self.greedy_vqe(
                ansatz_options, self.h_dvr_pd, vqe_options, log_file, prev_circs=best_circs
            )
            
            
            # Select the best circuits based on the energies from the current optimization
            inds = np.argsort(energies)
            best_circs = [best_circs[i] for i in inds[:ansatz_options['num_keep']]]
            best_energies = [energies[i] for i in inds[:ansatz_options['num_keep']]]
            best_params = [best_params[i] for i in inds[:ansatz_options['num_keep']]]
            best_converge_cnts = [converge_cnts[i] for i in inds[:ansatz_options['num_keep']]]
            best_converge_vals = [converge_vals[i] for i in inds[:ansatz_options['num_keep']]]
            best_ansatz_str = [ansatz_str[i] for i in inds[:ansatz_options['num_keep']]]
            if log_file:
                with open(log_file, 'a') as file:
                    file.write('*****************************************\n')
                    file.write(f'Layer {l+1} results:\n')
                    for i, energy in enumerate(best_energies):
                        file.write(f'Circuit {i+1}: Energy = {energy:.8f}\n')
                    file.write('*****************************************\n')
        
        # After completing all layers, select the overall best circuit
        ind = np.argmin(best_energies)
        best_overall_circuit = best_circs[ind]
        best_overall_energy = best_energies[ind]
        best_overall_params = best_params[ind]  
        best_overall_converge_cnts = best_converge_cnts[ind]
        best_overall_converge_vals = best_converge_vals[ind]
        best_ansatz_str = best_ansatz_str[ind]
        # Optionally log the final result
        if log_file:
            with open(log_file, 'a') as file:
                file.write(f'Final best circuit after {ansatz_options["layers"]} layers: {print(best_ansatz_str)}, Energy = {best_overall_energy:.8f}\n')
        
        print(f'Final best circuit after {ansatz_options["layers"]} layers')
        print(best_ansatz_str)
        print(f'Energy = {best_overall_energy:.8f}, Params = {best_overall_params:.8f}\n')
        return best_energies, best_circs, best_overall_circuit, best_overall_energy, best_overall_params, best_overall_converge_cnts, best_overall_converge_vals, best_ansatz_str

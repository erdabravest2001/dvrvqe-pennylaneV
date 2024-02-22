import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
import optax


class GreedyCircs:
    """
    Class representing a greedy search algorithm for building quantum circuits.
    """

    def __init__(self, ansatz_options, h_dvr_pd):
        self.ansatz_options = ansatz_options
        self.h_dvr_pd = h_dvr_pd  # Ensure this is a valid PennyLane observable
      

    def build_circuit(self, c, dev, ansatz_options):
        num_qubits = c.shape[0]  # Number of qubits in the circuit
        depth = c.shape[1]     # Depth of the circuit
        n_params = num_qubits
        params = np.zeros(n_params, requires_grad=True)
        @qml.qnode(dev)
        def circuit(params): 
            # Initialization based on ansatz options
            p = 0 
            if self.ansatz_options['add_h']:
                for i in range(num_qubits):
                    qml.Hadamard(wires=i)
            if self.ansatz_options['add_sx']:
                for i in range(num_qubits):
                    qml.SX(wires=i)
            if self.ansatz_options['add_rs']:
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
                        qml.ECR(wires=[q-1, q])
                    elif gate == 3:
                        qml.SX(wires=q)
                return qml.expval(self.h_dvr_pd)
        return circuit, params

    def get_greedy_ansatz(self, ansatz_options, best_circs, log_file=None):
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
            dev = qml.device('default.qubit', wires=ansatz_options['num_qubits'])
            ansatz, params = self.build_circuit(c, dev, ansatz_options)
            print("Circuit number: ", index)
            s = print(qml.draw(ansatz)(params))
            if (len(params) > 0) and (s not in ansatz_string_list):
                ansatz_list.append(ansatz)
                circ_list_final.append(c)
                ansatz_string_list.append(s)
                params_list.append(params)
                 
        return circ_list_final, ansatz_list, params_list, ansatz_string_list


    def opt_vqe(self, c, vqe_options, opt_params, log_file=None):
        dev = qml.device('default.qubit', wires=vqe_options['num_qubits'])

        @qml.qnode(dev)
        def cost_fn(params):
            return c(params)

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
    
    def greedy_vqe(self, ansatz_options, vqe_options, log_file=None, prev_circs=None):
        circ_list, ansatz_list, params_list, ansatz_string_list = self.get_greedy_ansatz(ansatz_options, prev_circs, log_file)
        best_params_list, best_energies_list, converge_cnts_list, converge_vals_list = [], [], [], []

        for i, (circ, params) in enumerate(zip(ansatz_list, params_list)):
            print(f"Optimizing Circuit {i+1}")
            # Initialize opt_params if needed, for example with zeros or a specific strategy
            initial_params = params_list[i]
            converge_cnts, converge_vals, optimized_params, best_energy = self.opt_vqe(c=circ, 
                                                                                        vqe_options=vqe_options, 
                                                                                        opt_params=initial_params,
                                                                                        log_file=log_file)

            best_params_list.append(optimized_params)
            best_energies_list.append(best_energy)
            converge_cnts_list.append(converge_cnts)
            converge_vals_list.append(converge_vals)

            if log_file:
                with open(log_file, 'a') as file:
                    file.write(f"Circuit {i+1}: Best Energy = {best_energy:.8f}, Counts = {len(converge_cnts)}\n")
            else:
                print(f"Circuit {i+1}: Best Energy = {best_energy:.8f}, Counts = {len(converge_cnts)}")

        # Process to select the best circuits based on optimized energies
        inds = np.argsort(best_energies_list)
        keep_inds = inds[:ansatz_options['num_keep']]
        best_circs = [circ_list[ind] for ind in keep_inds]
        best_params = [best_params_list[ind] for ind in keep_inds]
        best_energies = [best_energies_list[ind] for ind in keep_inds]
        converge_cnts = [converge_cnts_list[ind] for ind in keep_inds]
        converge_vals_list = [converge_vals_list[ind] for ind in keep_inds]

        return best_circs, converge_cnts, None, best_params, best_energies

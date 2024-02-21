import pennylane as qml
from pennylane import numpy as np
from qiskit.circuit import Parameter

class GreedyCircs:
    """
    Class representing a greedy search algorithm for building quantum circuits.
    """

    def __init__(self, ansatz_options, h_dvr_pd, num_qubits):
        self.ansatz_options = ansatz_options
        self.h_dvr_pd = h_dvr_pd  # Ensure this is a valid PennyLane observable
      

    def build_circuit(self, c, dev, ansatz_options):
        num_qubits = c.shape[0]  # Number of qubits in the circuit
        depth = c.shape[1]     # Depth of the circuit
        n_params = np.count_nonzero(c == 1) + (num_qubits if ansatz_options['add_rs'] else 0)
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
        return circuit, params, n_params

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
            dev = qml.device('lightning.qubit', wires=ansatz_options['num_qubits'])
            ansatz, params, nump= self.build_circuit(c, dev, ansatz_options)
            print("Circuit number: ", index)
            s = print(qml.draw(ansatz)(params))
            if (nump > 0) and (s not in ansatz_string_list):
                ansatz_list.append(ansatz)
                circ_list_final.append(c)
                ansatz_string_list.append(s)
                params_list.append(params)
                 
        return circ_list_final, ansatz_list, params_list, ansatz_string_list
    
    def greedy_vqe(self, ansatz_options, vqe_options, log_file=None, prev_circs=None):
        if prev_circs is None or len(prev_circs) == 0:
            pass

        circ_list, ansatz_list, params_list, ansatz_string_list = self.get_greedy_ansatz(ansatz_options, prev_circs, log_file) 
        best_params_list, best_energies_list, converge_cnts_list = [], [], []
        for i, (ansatz, params) in enumerate(zip(ansatz_list, params_list)):
            def cost_fn(params):
                return ansatz(params)
        
            optimizer = qml.GradientDescentOptimizer(stepsize=vqe_options['step_size'])
            current_params = None
            iteration_count = 0
            
            for iteration in range(vqe_options['max_iterations']):
                current_params = optimizer.step(cost_fn, current_params)
                current_energy = cost_fn(current_params)
                iteration_count += 1  # Increment the iteration count
                
                # Optional: Print/log the current iteration, energy, and parameters
                if iteration % 10 == 0:  # Adjust logging frequency as needed
                    print(f"Circuit {i+1}, Iteration {iteration+1}: Energy = {current_energy:.8f}, Counts = {iteration_count}")

            best_params_list.append(current_params)
            best_energies_list.append(current_energy)
            converge_cnts_list.append(iteration_count)  # Store the total count for this optimization
            
        # Select the best circuits based on the optimized energies
        inds = np.argsort(best_energies_list)
        keep_inds = inds[:ansatz_options['num_keep']]
        best_circs = [circ_list[ind] for ind in keep_inds]
        best_params = [best_params_list[ind] for ind in keep_inds]
        best_energies = [best_energies_list[ind] for ind in keep_inds]
        converge_cnts = [converge_cnts_list[ind] for ind in keep_inds]

        # Return the best circuits and their optimization results, including convergence counts
        return best_circs, converge_cnts, None, best_params, best_energies

import pennylane as qml
from pennylane import numpy as np

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


    def opt_vqe(self, h_dvr_pauli, ansatz, vqe_options, log_file=None, opt_params=None):
        # Ensure that the Hamiltonian is a PennyLane Hamiltonian
        
        # Define a device
        dev = qml.device('lightning.qubit', wires=vqe_options['num_qubits'])

        # Define the QNode for the VQE circuit
        @qml.qnode(dev)
        def circuit(params):
            ansatz(params)
            return qml.expval(h_dvr_pauli)

        # Define the cost function
        def cost_fn(params):
            return circuit(params)

        # Initialize optimizer
        optimizer_name = vqe_options['optimizer']
        stepsize = vqe_options.get('step_size', 0.01)  # Default step size
        if optimizer_name.lower() == "gradientdescent":
            optimizer = qml.GradientDescentOptimizer(stepsize)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported.")

        # Set the initial parameters
        if opt_params is None:
            params = np.random.rand(ansatz.num_params) * 2 * np.pi  # Random initial parameters
        else:
            params = opt_params

        counts, values = [], []
        best_params, best_energy = None, float('inf')

        # Optimization loop
        for i in range(vqe_options['max_iterations']):
            params, energy = optimizer.step_and_cost(cost_fn, params)

            counts.append(i)
            values.append(energy)

            if energy < best_energy:
                best_energy = energy
                best_params = params

            # Logging
            if i % 10 == 0 or i == vqe_options['max_iterations'] - 1:
                print(f"Iteration {i+1}: Energy = {energy:.8f}")

        return counts, values, best_params, best_energy

    def greedy_vqe(self, ansatz_options, vqe_options, log_file=None, prev_circs=None):
        # Get the greedy ansatz based on previous circuits and the current ansatz options
        circ_list, ansatz_list, params_list, ansatz_string_list = self.get_greedy_ansatz(ansatz_options, prev_circs, log_file)
        best_params_list, best_energies_list, converge_cnts_list = [], [], []

        for i, (ansatz, initial_param) in enumerate(zip(ansatz_list, params_list)):
            # Call opt_vqe for optimizing the current circuit
            converge_cnts, converge_vals, optimized_params, best_energy = self.opt_vqe(ansatz=ansatz, 
                                                                                        h_dvr_pauli=self.h_dvr_pd, 
                                                                                        vqe_options=vqe_options, 
                                                                                        log_file=log_file, 
                                                                                        opt_params=initial_param)

            # Store the results from opt_vqe
            best_params_list.append(optimized_params)
            best_energies_list.append(best_energy)
            converge_cnts_list.append(converge_cnts)

            # Optionally log the optimization result for the current circuit
            if log_file is not None:
                with open(log_file, 'a') as file:
                    file.write(f"Circuit {i+1}: Best Energy = {best_energy:.8f}, Counts = {len(converge_cnts)}\n")
            else:
                print(f"Circuit {i+1}: Best Energy = {best_energy:.8f}, Counts = {len(converge_cnts)}")

            # Select the best circuits based on optimized energies
            inds = np.argsort(best_energies_list)
            keep_inds = inds[:ansatz_options['num_keep']]
            best_circs = [circ_list[ind] for ind in keep_inds]
            best_params = [best_params_list[ind] for ind in keep_inds]
            best_energies = [best_energies_list[ind] for ind in keep_inds]
            converge_cnts = [converge_cnts_list[ind] for ind in keep_inds]

        return best_circs, converge_cnts, None, best_params, best_energies

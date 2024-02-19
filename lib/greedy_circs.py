import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

def build_circuit(c, ansatz_options, opt_level=2):    
    n_qubits = c.shape[0]  # Number of qubits in the circuit
    depth = c.shape[1]     # Depth of the circuit

    # Initialize a quantum circuit
    out = QuantumCircuit(n_qubits)

    # Calculate the number of parameters based on the input matrix 'c' and ansatz options
    n_params = np.count_nonzero(c == 1) + (n_qubits if ansatz_options['add_rs'] else 0)

    # Handling the case with no parameters
    if n_params == 0:
        theta = Parameter('X[0]')
        out.rz(theta, 0)

    # Initialize parameters for the gates
    thetas = [Parameter(f'X[{i}]') for i in range(n_params)]
    p = 0  # Parameter index

    # Track the last rotation on each qubit
    last_r = [False] * n_qubits

    # Add H gates if specified in ansatz options
    if ansatz_options['add_h']:
        for i in range(n_qubits):
            out.h(i)  
            last_r[i] = False 

    # Add SX gates if specified in ansatz options
    if ansatz_options['add_sx']:
        for i in range(n_qubits):
            out.sx(i)
            last_r[i] = False

    # Add Rz gates if specified in ansatz options
    if ansatz_options['add_rs']:
        for i in range(n_qubits):
            out.rz(thetas[p], i)
            last_r[i] = False
            p += 1
    
    # Iterate through the depth of the circuit
    for i in range(depth):
        layer = c[:, i]  # Get the i-th layer of gates
        for q, gate in enumerate(layer):
            if gate == 0:
                pass  # Do nothing for gate 0
            elif gate == 1 and not last_r[q]:
                out.rz(thetas[p], q)  # Apply Rz gate
                last_r[q] = True
                p += 1
            elif gate == 2:
                q_new = q - (gate - 1)
                out.ecr(q, q_new)  # Apply ECR gate
                last_r[q] = False
                last_r[q_new] = False
            elif gate == 3:
                out.sx(q)  # Apply SX gate
                last_r[q] = False
                last_r[q - (gate - 1)] = False

    # Transpile the circuit for optimization
    out = transpile(out, optimization_level=opt_level)
    return out


def find_allowed_gates(num_qubits, reps, partitions):
    import numpy as np

    tri_inds = np.triu_indices(num_qubits, k=1)
    num_gates = len(tri_inds[0])
    
    allowed_gates = []
    for i in range(reps):
        for j, (q1, q2) in enumerate(zip(*tri_inds)):
            ind1 = [k for k, el in enumerate(partitions) if q1 in el]
            ind2 = [k for k, el in enumerate(partitions) if q2 in el]
            if ind1[0] == ind2[0]:
                allowed_gates.append(j + i * num_gates)
    return allowed_gates

def partition_string_to_list(p_string):
    p_list = p_string.split('-')
    out = list(map(lambda p: [int(s) for s in p], p_list))
    return out

def partition_list_to_string(p_list):
    out = list(map(lambda p: ''.join(map(lambda i: str(i), p)), p_list))
    out = '-'.join(out)
    return out

def build_circuit_ent(ansatz_options, gates, simplify=False):
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    num_qubits = ansatz_options['num_qubits']
    reps = ansatz_options['reps']
    constructive = ansatz_options['constructive']
    ansatz = QuantumCircuit(num_qubits)
    ansatz.global_phase = 3*np.pi/2
    tri_inds = np.triu_indices(num_qubits, k=1)
    num_gates = len(tri_inds[0])
    p = 0
    prev_rots = []
    for i in range(reps):
        for q in range(num_qubits):
            if (not simplify) or ((simplify) and (q not in prev_rots)):
                theta = Parameter(f'$x_{{{p}}}$')
                p += 1
                ansatz.rz(theta, q)
                ansatz.x(q)
                ansatz.rz(3*np.pi/2,q)
                prev_rots.append(q)
        ansatz.barrier(range(num_qubits))
        for j, (q1, q2) in enumerate(zip(*tri_inds)):
            if constructive:
                if j + i * num_gates in gates:
                    ansatz.ecr(q1, q2)
                    if q1 in prev_rots:
                        prev_rots.remove(q1)
                    if q2 in prev_rots:
                        prev_rots.remove(q2)
            else: 
                if j + i * num_gates not in gates:
                    ansatz.ecr(q1, q2)
                    if q1 in prev_rots:
                        prev_rots.remove(q1)
                    if q2 in prev_rots:
                        prev_rots.remove(q2)
        ansatz.barrier(range(num_qubits))
    for q in range(num_qubits):
        if (not simplify) or ((simplify) and (q not in prev_rots)):
            theta = Parameter(f'$x_{{{p}}}$')
            # theta = Parameter(f'')
            p += 1
            ansatz.rz(theta, q)
            ansatz.x(q)
            prev_rots.append(q)
    return ansatz

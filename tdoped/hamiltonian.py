import numpy as np
from typing import Union
import qiskit
from qiskit.quantum_info import Clifford, Pauli, PauliList

class HamiltonianTools:
    @staticmethod
    def map_to_z(pauli: Pauli, target: int = 0) -> Clifford:
        """
        Map a given Pauli operator to the Z basis via Clifford operations.
        This method constructs a Clifford circuit that maps the specified Pauli 
        operator to the Z basis. The mapping is achieved using a series of 
        Hadamard (H), S-dagger (S†), and CNOT (CX) gates. If the target qubit 
        is initially an identity ('I'), the method selects a new target qubit 
        that is not an identity.
        Args:
            pauli (Pauli): The Pauli operator to be mapped to the Z basis.
            target (int, optional): The target qubit index. Defaults to 0.
        Returns:
            Clifford: The Clifford operator that maps the Pauli operator to the Z basis.
        """
        N = pauli.num_qubits
        qc = qiskit.QuantumCircuit(N)
        pauli_str = pauli.to_label().replace('-','')
        
        if pauli_str[target] == 'I':
            old_target = target
            target = min(i for i in range(N) if pauli_str[i] != 'I')
        else:
            old_target = None
            
        for i in range(N):
            if pauli_str[i] == 'X':
                qc.h(i)
            elif pauli_str[i] == 'Y':
                qc.sdg(i)
                qc.h(i)
                
        for i in range(N):
            if i != target and pauli_str[i] != 'I':
                qc.cx(i, target)
                
        cliff = Clifford.from_circuit(qc.reverse_bits()).adjoint()
        
        if old_target is not None:
            swap_qc = qiskit.QuantumCircuit(N)
            swap_qc.swap(target, old_target)
            cliff = Clifford.from_circuit(swap_qc.reverse_bits()).compose(cliff)
            
        return cliff
    
    @staticmethod
    def reduced_p(paulis: Union[Pauli, PauliList], config: np.ndarray) -> Pauli:
        """Get reduced Pauli operator for given configuration.
        
        Args:
            pauli: Full Pauli operator
            config: Binary configuration for reduced system
            
        Returns:
            Reduced Pauli operator
        """
        def reduced_p_single(pauli):
            label = pauli.to_label().replace('-', '')
            n_full = len(label)
            n_reduced = len(config)
            if any([char == 'X' or char == 'Y' for char in label[:n_reduced]]):
                ret_dim = 2**(n_full - n_reduced)
                return np.zeros((ret_dim, ret_dim))
            
            # Get sign from configuration
            sign = 1
            if n_reduced > 0:
                for i in range(n_reduced):
                    if label[i] == 'Z' and config[i] == 1:
                        sign *= -1
                        
            # Get reduced operator
            if n_reduced < n_full:
                reduced_label = label[n_reduced:]
            else:
                reduced_label = 'I' * (n_reduced - n_full)
            return sign * Pauli(reduced_label).to_matrix()
        
        
        if isinstance(paulis, Pauli): return reduced_p_single(paulis)
        else: return np.array([reduced_p_single(p) for p in paulis])
    
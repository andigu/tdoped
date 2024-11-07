import numpy as np
from qiskit.quantum_info import Pauli, PauliList, Clifford, PauliList
from tqdm.auto import *
import qiskit

class StabilizerOps:
    @staticmethod
    def rref_bin(A: np.ndarray) -> np.ndarray:
        """
        Compute the Reduced Row Echelon Form (RREF) of a binary matrix over GF(2).

        Parameters:
        A (np.ndarray): A binary matrix (elements are 0 or 1) of shape (m, n).

        Returns:
        np.ndarray: The RREF of the input matrix over GF(2).

        Notes:
        - The function operates on a copy of the input matrix and does not modify the original matrix.
        - The arithmetic is performed in the Galois Field GF(2), meaning all operations are modulo 2.
        """
        A = A.copy() % 2
        rows, cols = A.shape
        row = 0
        for col in range(cols):
            if row >= rows: break
            pivot_row = np.where(A[row:, col] == 1)[0]
            if len(pivot_row) == 0: continue
            pivot_row += row
            A[[row, pivot_row[0]]] = A[[pivot_row[0], row]]
            for r in range(rows):
                if r != row and A[r, col] == 1:
                    A[r] ^= A[row]
            row += 1
        return A

    @staticmethod
    def null_space_basis(A: np.ndarray) -> np.ndarray:
        """
        Compute the null space basis of a given binary matrix A.
        This function calculates the null space basis of a binary matrix A using
        the reduced row echelon form (RREF). It identifies the pivot and free columns
        to construct the basis vectors of the null space. The resulting basis is
        verified to ensure it provides a commuting basis.
        Parameters:
        A (np.ndarray): A binary matrix for which the null space basis is to be computed.
        Returns:
        np.ndarray: An array containing the basis vectors of the null space of A.
        Raises:
        AssertionError: If the resulting basis does not provide a commuting basis.
        """
        R = StabilizerOps.rref_bin(A)
        rows, cols = R.shape
        pivot_cols = [np.where(R[r])[0][0] for r in range(rows) if np.any(R[r])]
        free_cols = [c for c in range(cols) if c not in pivot_cols]
        
        basis = []
        for free_col in free_cols:
            vec = np.zeros(cols, dtype=int)
            vec[free_col] = 1
            for r, pivot_col in enumerate(pivot_cols):
                if pivot_col >= cols: break
                if R[r, free_col] == 1:
                    vec[pivot_col] = 1
            basis.append(vec)
        nsp = np.array(basis)
        assert np.all((StabilizerOps.swap_xz(nsp) @ nsp.T)%2 == 0) # Ensure we provide a commuting basis
        return nsp

    @staticmethod
    def to_pauli(stabilizers: np.ndarray) -> PauliList:
        """Convert symplectic representation to PauliList."""
        N = stabilizers.shape[1]//2
        return PauliList([Pauli((x[N:], x[:N])) for x in stabilizers])

    @staticmethod
    def swap_xz(A: np.ndarray) -> np.ndarray:
        """Swap X and Z components of Pauli array."""
        _, N2 = A.shape
        N = N2//2
        return np.roll(A, N, axis=1)
    
    @staticmethod
    def get_cliff(stabilizers):
        """
        Generate a Clifford operator that maps the given stabilizers to the Z-basis.

        Args:
            stabilizers (list): A list of stabilizer generators.

        Returns:
            Clifford: A Clifford operator that maps the given stabilizers to the Z-basis.
        """
        paulis = StabilizerOps.to_pauli(stabilizers)
        cliff = StabilizerOps.map_to_z(paulis[0], target=0)
        for i in trange(1, len(paulis)):
            curr_paulis = [p.evolve(cliff) for p in paulis]
            p_str = list(curr_paulis[i].to_label().replace('-', ''))
            for j in range(i):
                assert p_str[j] == 'Z' or p_str[j] == 'I'
                if p_str[j] == 'Z':
                    p_str[j] = 'I'
            p_str = ''.join(p_str)
            cliff = StabilizerOps.map_to_z(Pauli(p_str), target=i).compose(cliff)
        return cliff
    
    @staticmethod
    def map_to_z(pauli, target=0):
        """
        Maps a given Pauli operator to the Z-basis using a quantum circuit.
        Args:
            pauli (qiskit.quantum_info.Pauli): The Pauli operator to be mapped.
            target (int, optional): The target qubit index to map to the Z-basis. Defaults to 0.
        Returns:
            qiskit.quantum_info.Clifford: The Clifford operator representing the adjoint of the circuit 
            that maps the given Pauli operator to the Z-basis.
        """
        N = pauli.num_qubits
        qc = qiskit.QuantumCircuit(N)
        pauli = pauli.to_label().replace('-','')
        old_target = None
        if pauli[target] == 'I':
            old_target, target = target, min(i for i in range(N) if pauli[i] != 'I')
        for i in range(N):
            if pauli[i] == 'X':
                qc.h(i)
            if pauli[i] == 'Y':
                qc.sdg(i)
                qc.h(i)
                
        for i in range(N):
            if i != target and pauli[i] != 'I':
                qc.cx(i, target)
        qc = qc.reverse_bits()
        ret = Clifford.from_circuit(qc).adjoint()
        if old_target is not None:
            qc2 = qiskit.QuantumCircuit(N)
            qc2.swap(target, old_target)
            qc2 = qc2.reverse_bits()
            ret = Clifford.from_circuit(qc2).compose(ret)
        return ret
    
    @staticmethod
    def get_diagonalizing_clifford(paulis):
        """
        Computes the Clifford transformation that diagonalizes a given set of Pauli operators.

        Args:
            paulis (list): A list of Pauli operators to be diagonalized.

        Returns:
            Clifford: The Clifford transformation that diagonalizes the input Pauli operators.
        """
        nsp = StabilizerOps.null_space_basis(StabilizerOps.swap_xz(paulis)) # nsp is stab group for new eigenstates
        return StabilizerOps.get_cliff(nsp)
        
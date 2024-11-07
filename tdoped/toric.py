import numpy as np
from qiskit.quantum_info import Pauli, PauliList, StabilizerState, PauliList, Statevector
from tdoped.stabilizer import StabilizerOps
import itertools as it

class ToricCode:
    def __init__(self, size: int, perturbation):
        """
        Initialize a 2D toric code with a given size and perturbation.

        Args:
            size (int): The size of the toric code.
            perturbation: Symplectic representation of perturbing Paulis.

        Attributes:
            N (int): The size of the toric code.
            num_qubits (int): The number of qubits in the toric code.
            perturbation: Symplectic representation of perturbing Paulis.
            stabilizers: The stabilizers of the toric code.
            diagonalizing_cliff: The Clifford transformation that diagonalizes the remaining stabilizers of the perturbed model.
            transformed_stab: The transformed stabilizers after applying the Clifford transformation.
            transformed_pert: The transformed perturbation after applying the Clifford transformation.
            S (int): The number of remaining stabilizers for the perturbed toric code.
            subsys_size (int): The size of the non-stabilizer.
            hamiltonian_data (tuple): A tuple containing the sign matrix, phases, and Pauli matrices.
        """
        self.N = size
        self.num_qubits = 2 * size * size
        self.perturbation = perturbation # Symplectic representation of perturbing Paulis
        self.stabilizers = self.get_stabilizers()
        full_group = np.concatenate([self.stabilizers, perturbation], axis=0)
        self.diagonalizing_cliff = StabilizerOps.get_diagonalizing_clifford(full_group)

        self.transformed_stab = StabilizerOps.to_pauli(self.stabilizers).evolve(self.diagonalizing_cliff)
        self.transformed_pert = StabilizerOps.to_pauli(self.perturbation).evolve(self.diagonalizing_cliff)

        def string_index(st, char):
            idx = st.to_label().replace('-', '').find(char)
            return idx if idx>=0 else np.inf
        self.S = min([min(string_index(x, 'X'), string_index(x, 'Y')) for x in self.transformed_pert+self.transformed_stab])
        self.subsys_size = 2*self.N**2 - self.S

        sign_mat = np.array([np.array(list(p.to_label().replace('-', '')[:self.S])) == 'Z' 
                     for p in self.transformed_stab+self.transformed_pert]).astype(float)
        phases = np.array([p.phase for p in self.transformed_stab+self.transformed_pert]).astype(float)
        paulis = np.array([Pauli(p.to_label().replace('-', '')[self.S:]).to_matrix() 
                        for p in self.transformed_stab+self.transformed_pert])
        self.hamiltonian_data = (sign_mat, phases, paulis)

    @staticmethod
    def coordinate(N: int, i: int, j: int, xy: int) -> int:
        """
        Calculate the coordinate in a toric grid.

        Args:
            N (int): The size of the grid (NxN).
            i (int): The row index.
            j (int): The column index.
            xy (int): The plane index (0 for x-plane, 1 for y-plane).

        Returns:
            int: The calculated coordinate in the grid.
        """
        return xy*N**2 + N*i + j
        
    def get_stabilizers(self) -> np.ndarray:
        """
        Generate the stabilizer generators for the toric code.
        The method constructs both star and plaquette operators for each site 
        on the toric code lattice, as well as the logical operators. The 
        stabilizers are represented as concatenated arrays of X and Z operators.
        Returns:
            np.ndarray: An array of stabilizer generators, where each stabilizer 
            is represented as a concatenated array of X and Z operators.
        """
        stabilizers = []
        # Star operators
        for i in range(self.N):
            for j in range(self.N):
                X, Z = np.zeros((2, 2*self.N**2))
                X[ToricCode.coordinate(self.N,i,j,0)] = 1
                X[ToricCode.coordinate(self.N,i,(j+1)%self.N,0)] = 1
                X[ToricCode.coordinate(self.N,i,j,1)] = 1
                X[ToricCode.coordinate(self.N,(i+1)%self.N,j,1)] = 1
                stabilizers.append(np.concatenate([X,Z]))
                
                # Plaquette operators
                X, Z = np.zeros((2, 2*self.N**2))
                Z[ToricCode.coordinate(self.N,i,j,0)] = 1
                Z[ToricCode.coordinate(self.N,i,j,1)] = 1
                Z[ToricCode.coordinate(self.N,(i-1)%self.N,j,0)] = 1
                Z[ToricCode.coordinate(self.N,i,(j-1)%self.N,1)] = 1
                stabilizers.append(np.concatenate([X,Z]))
        
        # Logical operators
        X, Z = np.zeros((2, 2*self.N**2))
        for i in range(self.N):
            Z[ToricCode.coordinate(self.N,i,0,1)] = 1
        stabilizers.append(np.concatenate([X,Z]))
        
        X, Z = np.zeros((2, 2*self.N**2))
        for i in range(self.N):
            Z[ToricCode.coordinate(self.N,0,i,0)] = 1
        stabilizers.append(np.concatenate([X,Z]))
        return np.array(stabilizers).astype(int)

    @property
    def unperturbed_ground_state(self) :
        """
        Compute the unperturbed ground state of the system.
        """
        tmp = np.stack([self.transformed_stab.x, self.transformed_stab.z]).transpose((1,2,0)).reshape((-1, 4*self.N**2))
        assert np.all(tmp[:,::2] == self.transformed_stab.x) and np.all(tmp[:,1::2] == self.transformed_stab.z)
        ptraced = StabilizerOps.rref_bin(tmp)[:,:2*self.subsys_size]
        ptraced = np.array([x for x in ptraced if not np.all(x==0)])
        X, Z = ptraced[:,::2], ptraced[:,1::2]
        
        state = StabilizerState.from_stabilizer_list(PauliList([Pauli((Z[i], X[i])) for i in range(len(X))]))
        return np.array(Statevector(state.clifford.to_circuit()))
    
    def get_sample_subsys_observables(self):
        """
        Returns 4^6 observables for hexagonal shaped subsystem of size 6
        """
        observables = []
        coordinates = [(0,0,0), (0,0,1), (0,1,1), (0,2,0), (1,1,1), (1,0,1)]
        for pauli in it.product(["I", "X", "Y", "Z"], repeat=6):
            X, Z = np.zeros((2,2*self.N**2))
            for pos, char in enumerate(pauli):
                if char in ['Y', 'Z']: Z[ToricCode.coordinate(self.N, *coordinates[pos])] = 1
                if char in ['X', 'Y']: X[ToricCode.coordinate(self.N, *coordinates[pos])] = 1
            observables.append(np.concatenate([X,Z]))
        observables = np.array(observables)
        transformed_obs = StabilizerOps.to_pauli(observables).evolve(self.diagonalizing_cliff)
        return observables, transformed_obs

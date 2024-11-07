import numpy as np
from typing import List
import scipy.sparse.linalg
from numba import njit
from tqdm.auto import *
import jax
import jax.numpy as jnp

class Simulator:
    """
    Simulator class for performing various quantum simulations.
    """

    @staticmethod
    def time_evolve(initial_state: np.ndarray,
                   hamiltonian: np.ndarray,
                   t0: float,
                   tf: float,
                   num: int,
                   observables: List[np.ndarray], pbar: bool = False) -> np.ndarray:
        """
        Simulate time evolution under block-diagonalized Hamiltonian.

        Args:
            initial_state (np.ndarray): The initial state vector of the system.
            hamiltonian (np.ndarray): The Hamiltonian matrix governing the time evolution.
            t0 (float): The initial time.
            tf (float): The final time.
            num (int): The number of time points to evaluate.
            observables (List[np.ndarray]): A list of observable matrices to measure at each time point.
            pbar (bool, optional): If True, display a progress bar. Defaults to False.

        Returns:
            np.ndarray: A 2D array where each row corresponds to the time evolution of an observable.
        """
        results = []
        for observable in (tqdm(observables) if pbar else observables):
            if np.all(np.isclose(observable, 0)): results.append(np.zeros((num,)))
            else:
                results.append([np.einsum('i,ij,j', np.conj(x), observable, x) 
                                for x in scipy.sparse.linalg.expm_multiply(
                                    -1j*hamiltonian, initial_state, start=t0, stop=tf, num=num)])
        return np.array(results)
    

    def gibbs_sample(hamiltonian_data, all_coeffs, S, beta=1., copies=10, samples=100):
        """
        Perform Gibbs sampling to generate samples from a given Hamiltonian.
        Parameters:
        -----------
        hamiltonian_data : tuple
            A tuple containing the sign matrix, phases, and Pauli matrices.
        all_coeffs : array-like
            Coefficients for the Hamiltonian terms.
        S : int
            The number of remaining stabilizers.
        beta : float, optional
            Inverse temperature parameter (default is 1.0).
        copies : int, optional
            Number of copies of the system to sample in parallel (default is 10).
        samples : int, optional
            Number of samples to generate (default is 100).
        Returns:
        --------
        np.ndarray
            An array of shape (samples, copies, S) containing the generated samples.
        Notes:
        ------
        This function uses JAX for just-in-time compilation and efficient computation.
        """
        @jax.jit
        def Z(x, beta, sign_mat, phases, all_coeffs, paulis):
            """
            Compute the partition function for a given configuration x.
            Parameters:
            -----------
            x : jnp.ndarray
                Current configuration of spins (i.e., signs of the stabilizers).
            beta : float
                Inverse temperature parameter.
            sign_mat : jnp.ndarray
                Sign matrix from the Hamiltonian data.
            phases : jnp.ndarray
                Phases from the Hamiltonian data.
            all_coeffs : jnp.ndarray
                Coefficients for the Hamiltonian terms.
            paulis : jnp.ndarray
                Pauli matrices from the Hamiltonian data.
            Returns:
            --------
            jnp.ndarray
                Log of the partition function for each copy.
            """
            coeffs = (-1)**(sign_mat@x.T + phases[:,None]) * all_coeffs[:,None]
            h = jnp.einsum('ij,ikl', coeffs, paulis)
            eners = jnp.linalg.eigvalsh(h)
            return jax.scipy.special.logsumexp(-beta*eners, axis=1)
        
        sign_mat, phases, paulis = [jnp.array(x) for x in hamiltonian_data]
        all_coeffs = jnp.array(all_coeffs)
        x = np.random.choice(2, (copies,S), p=[0.9,0.1])
        hist = []
        rands = np.random.rand(samples, copies, S)
        for i in trange(samples):
            hist.append(np.copy(x))    
            for j in range(S):
                x[:,j] = 0
                Z0 = Z(x, beta, sign_mat, phases, all_coeffs, paulis)
                x[:,j] = 1
                Z1 = Z(x, beta, sign_mat, phases, all_coeffs, paulis)
                p = scipy.special.softmax(np.stack([Z0,Z1]), axis=0)
                
                new_config = rands[i,:,j] < p[1]
                x[:,j] = new_config
        return np.array(hist)
    
    @staticmethod
    def reduced_ham(hamiltonian_data, all_coeffs, x):
        """
        Compute the reduced Hamiltonian matrix.

        Parameters:
        -----------
        hamiltonian_data : tuple
            A tuple containing:
            - sign_mat (np.ndarray): A matrix of signs for each of the Paulis in the Hamiltonian.
            - phases (np.ndarray): A vector of phase values.
            - paulis (np.ndarray): A tensor of Pauli matrices.
        all_coeffs : np.ndarray
            An array of coefficients.
        x : np.ndarray
            The symmetry sector for which we want to evaluate the Hamiltonian (i.e., the stabilizer signs).

        Returns:
        --------
        np.ndarray
            The reduced Hamiltonian matrix.
        """
        sign_mat, phases, paulis = hamiltonian_data
        coeffs = (-1)**(sign_mat@x.T + phases) * all_coeffs        
        return np.einsum('i,ikl', coeffs, paulis)
    
    def ground_state(hamiltonian_data, all_coeffs, S, copies=10, T_start=1.0, T_end=1e-6, steps=1000):
        """
        Perform simulated annealing to find the ground state of a Hamiltonian.
        Parameters:
        -----------
        hamiltonian_data : tuple
            A tuple containing the sign matrix, phases, and Pauli matrices.
        all_coeffs : array-like
            Coefficients for the Hamiltonian terms.
        S : int
            The number of spins or qubits.
        copies : int, optional
            The number of copies or parallel simulations to run (default is 10).
        T_start : float, optional
            The starting temperature for annealing (default is 1.0).
        T_end : float, optional
            The ending temperature for annealing (default is 1e-6).
        steps : int, optional
            The number of annealing steps (default is 1000).
        Returns:
        --------
        best_x : ndarray
            The configuration of stabilizer signs with the lowest energy.
        best_E : float
            The lowest energy found during the annealing process.
        """
        @jax.jit
        def E0(x, sign_mat, phases, all_coeffs, paulis):
            coeffs = (-1)**(sign_mat@x.T + phases[:,None]) * all_coeffs[:,None]
            h = jnp.einsum('ij,ikl', coeffs, paulis)
            eners = jnp.linalg.eigvalsh(h)
            return jnp.min(eners, axis=-1)
        
        sign_mat, phases, paulis = [jnp.array(x) for x in hamiltonian_data]
        all_coeffs = jnp.array(all_coeffs)
        current_x = np.random.choice(2, size=(copies-1,S))
        current_x = np.concatenate([current_x, np.zeros((1,S))], axis=0).astype(bool)
        current_E = E0(current_x, sign_mat, phases, all_coeffs, paulis)
        
        # Track best solution
        best_x = current_x.copy()
        best_E = current_E
        
        # Annealing schedule
        T = np.geomspace(T_start, T_end, steps)
        
        for t in range(steps):
            # Propose new state by flipping random bits
            flip_mask = np.random.random((copies, S)) < 0.05  # 10% flip probability
            proposed_x = current_x ^ flip_mask
            
            # Calculate energies
            proposed_E = E0(proposed_x, sign_mat, phases, all_coeffs, paulis)
            
            # Metropolis acceptance criterion
            dE = proposed_E - current_E
            accept_prob = np.exp(np.clip(-dE / T[t], -100, 0))
            accept = np.random.random(copies) < accept_prob
            
            # Update states
            current_x[accept] = proposed_x[accept]
            current_E = E0(current_x, sign_mat, phases, all_coeffs, paulis)
            
            # Update best solution
            improved = current_E < best_E
            best_x[improved] = np.copy(current_x[improved])
            best_E = np.minimum(best_E, current_E)
        
        # Return best configuration and corresponding energy
        return best_x[np.argmin(best_E)], np.min(best_E)


    
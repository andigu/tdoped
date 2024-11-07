# tdoped

`tdoped` is a Python package designed for the simulation and analysis of stabilizer Hamiltonians with perturbations. The theoretical foundation of this work is detailed in our [paper](https://arxiv.org/abs/2403.14912).

## Features

- **Simulation of Toric Codes**: Initialize and manipulate 2D toric codes with various sizes and perturbations.
- **Perturbation Handling**: Handle various types of perturbations in the toric code, including local and global perturbations.
- **Ground State Calculation**: Compute the ground state of the perturbed Hamiltonian using simulated annealing.
- **Time Evolution**: Simulate the time evolution of the system under the perturbed Hamiltonian.
- **Gibbs Sampling**: Perform Gibbs sampling to generate samples from perturbed stabilizer Hamiltonians.
- **Topological Entanglement Entropy**: Calculate the topological entanglement entropy of the ground state of perturbed stabilizer Hamiltonians.

## Simulations

### Perturbed Toric Codes

The `tdoped` package allows for the simulation of toric codes with various perturbations. Perturbations can be specified in the symplectic representation and can include either local perturbations (i.e., only affecting individual qubits or small groups of qubits), or global perturbations that affect the entire system or large regions of the system. Runtimes of each subroutine generally scale exponentially in the *number* of Pauli perturbations added, independent of the strength or locality of these perturbations.

For usage of this package, please refer to the `example.ipynb` notebook included in the repository.
# -*- coding: utf-8 -*-

import numpy as np
from data import *
from vqc import *
from qiskit import circuit
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
from qiskit.quantum_info.operators import Operator
from qiskit.opflow import I, X, Y, Z


class Partition2(VQC):
    """Implement circuit partition strategy in paper: Constructing a Virtual Two-qubit Gate by Sampling Single-qubit Operations."""
    def __init__(self, data, n_qubits=2, n_bits=2, feature_reps=1, paulis=['Z', 'ZZ'], ansatz_reps=1, n_shots=100, lr=0.1, epochs=3):
        super().__init__(data, n_qubits, n_bits, feature_reps, paulis, ansatz_reps, n_shots, lr, epochs)

    def initialization(self):
        """Initialize the quantum circuit."""
        # Initialize the quantum circuit
        self.pauli_feature_map = self.set_feature_map()
#       self.ansatz = self.set_ansatz()
#       small_circuit1 = self.construct_circuit(plot=True)
#       small_circuit2 = self.construct_circuit(plot=True)
#       self.circuit = [small_circuit1, small_circuit2]

#       # Initialize parameters
#       params1 = np.random.random((self.L, self.ansatz.num_parameters))
#       params2 = np.random.random((self.L, self.ansatz.num_parameters))
#       weights = np.random.random(2 * self.L)
#       self.params = [params1, params2, weights]

    def set_ansatz(self, switch=False, plot=False):
        """Ansatz circuit."""
        ansatz = QuantumCircuit(2)
        ansatz.rz(circuit.Parameter('theta0'), 0)
        ansatz.rz(circuit.Parameter('theta1'), 1)
        ansatz.ry(circuit.Parameter('theta2'), 0)
        ansatz.ry(circuit.Parameter('theta3'), 1)
        ansatz.cz(0, 1)
        ansatz.z(0)
        ansatz.z(1)

        space = [-1, 1]
        alpha1 = np.random.choice(space, 1)
        alpha2 = np.random.choice(space, 1)
        project = Operator((I + alpha2 * Z) / 2)
        if switch:
            ansatz.append(project, 0)
            ansatz.append(project, 1)
            ansatz.rz(alph1 * np.pi / 2, 0)
            ansatz.rz(alph1 * np.pi / 2, 1)
        else:
            ansatz.rz(alph1 * np.pi / 2, 0)
            ansatz.rz(alph1 * np.pi / 2, 1)
            ansatz.append(project, 0)
            ansatz.append(project, 1)
        if plot:
            ansatz.draw(output="mpl")
            plt.show()
        return ansatz


model = Partition2(digits(method='svd', n_components=4), n_qubits=2, n_bits=2)
model.set_ansatz(plot=True)

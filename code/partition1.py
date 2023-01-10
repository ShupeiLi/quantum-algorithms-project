# -*- coding: utf-8 -*-

from data import *
from vqc import *
from qiskit.circuit.library import TwoLocal
from qiskit import circuit


class Partition1(VQC):
    """Implement circuit partition strategy in paper: High Dimensional Quantum Machine Learning With Small Quantum Computers. (arXiv:2203.13739v)"""
    def __init__(self, data, n_qubits=2, n_bits=2, feature_reps=1, paulis=['Z', 'ZZ'], ansatz_reps=1, n_shots=100, lr=0.1, epochs=3, L=1):
        self.L = L
        super().__init__(data, n_qubits, n_bits, feature_reps, paulis, ansatz_reps, n_shots, lr, epochs)

    def initialization(self):
        """Initialize the quantum circuit."""
        # Initialize the quantum circuit
        self.pauli_feature_map = self.set_feature_map()
        self.ansatz = self.set_ansatz()
        small_circuit1 = self.construct_circuit(plot=False)
        small_circuit2 = self.construct_circuit(plot=False)
        self.circuit = [small_circuit1, small_circuit2]

        # Initialize parameters
        params1 = np.random.random((self.L, self.ansatz.num_parameters))
        params2 = np.random.random((self.L, self.ansatz.num_parameters))
        weights = np.random.random(2 * self.L)
        self.params = [params1, params2, weights]

    def set_ansatz(self, plot=False):
        """Ansatz circuit."""
        ansatz = TwoLocal(
                num_qubits=self.n_qubits,
                rotation_blocks=['rz', 'ry'],
                entanglement_blocks='cz',
                reps=self.ansatz_reps).decompose()
        ansatz.rz(circuit.Parameter('psi0'), 0)
        ansatz.rz(circuit.Parameter('psi1'), 1)
        if plot:
            ansatz.draw(output="mpl")
            plt.show()
        return ansatz

    def predict(self, _circuit, X, params):
        """Predict one record."""
        result1 = list()
        result2 = list()
        for i in range(self.L):
            result1.append(super().predict(_circuit[0], X, params[0][i, :]))
            result2.append(super().predict(_circuit[1], X, params[1][i, :]))
        return np.sum(params[2] * np.array(result1 + result2))

    def gradient_descent(self, _circuit, X, y, params):
        """Optimize the parameter with Gradient Descent algorithm."""
        def loss(_circuit, X, y, params):
            return np.abs(self.predict(_circuit, X, params) - y)

        def subroutine(_circuit, X, y, index_i, index_j, params, l=-1):
            if l != -1:
                params_target = params[index_i][l, :]
            else:
                params_target = params[index_i]
            e_i = np.identity(params_target.size)[:, index_j]
            plus = params_target + (np.pi / 2) * e_i
            minus = params_target - (np.pi / 2) * e_i
            plus_lst = list()
            minus_lst = list()
            for k in range(len(params)):
                if k == index_i:
                    if l != -1:
                        plus_arr = params[index_i].copy()
                        plus_arr[l, :] = plus
                        minus_arr = params[index_i].copy()
                        minus_arr[l, :] = minus
                        plus_lst.append(plus_arr)
                        minus_lst.append(minus_arr)
                    else:
                        plus_lst.append(plus)
                        minus_lst.append(minus)
                else:
                    plus_lst.append(params[k])
                    minus_lst.append(params[k])
            return (loss(_circuit, X, y, plus_lst) - loss(_circuit, X, y, minus_lst)) / 2

        for i in range(len(params)):
            if i < 2:
                for l in range(self.L):
                    for j in range(len(params[i])):
                        params[i][l, j] -= self.lr * subroutine(_circuit, X, y, i, j, params, l)
            else:
                for j in range(len(params[i])):
                    params[i][j] -= self.lr * subroutine(_circuit, X, y, i, j, params, -1)

        return params


def five_fold_cross_validation(data, n_qubits=2, n_bits=2):
    cv_lst = cross_validation_split(data, n_folds=5)
    for i in range(len(cv_lst)):
        model = Partition1(cv_lst[i], n_qubits=n_qubits, n_bits=n_bits, epochs=3, lr=0.1, L=2, n_shots=1000)
        model.main()


if __name__ == '__main__':
    # digits
    data = digits(method='svd', n_components=4)
    five_fold_cross_validation(data)

    # breast_cancer
    data = cancer(method='svd', n_components=4)
    five_fold_cross_validation(data)

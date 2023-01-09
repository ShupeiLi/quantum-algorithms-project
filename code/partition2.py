# -*- coding: utf-8 -*-

from data import *
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

dev = qml.device('default.qubit', wires=4)
is_cut = True

def statepreparation(x):
    qml.Hadamard(wires=0)
    qml.PauliRot(2 * x[0], 'Z',  wires=0)
    qml.Hadamard(wires=1)
    qml.PauliRot(2 * x[1], 'Z',  wires=1)
    qml.Hadamard(wires=2)
    qml.PauliRot(2 * x[2], 'Z',  wires=2)
    qml.Hadamard(wires=3)
    qml.PauliRot(2 * x[3], 'Z',  wires=3)

    qml.CNOT(wires=[0, 1])
    qml.PauliRot(2 * (np.pi - x[0]) * (np.pi - x[1]), 'Z',  wires=1)
    qml.CNOT(wires=[0, 1])

    if not is_cut:
        qml.CNOT(wires=[0, 2])
        qml.PauliRot(2 * (np.pi - x[0]) * (np.pi - x[2]), 'Z',  wires=2)
        qml.CNOT(wires=[0, 2])

    qml.CNOT(wires=[2, 3])
    qml.PauliRot(2 * (np.pi - x[2]) * (np.pi - x[3]), 'Z',  wires=3)
    qml.CNOT(wires=[2, 3])

    qml.CNOT(wires=[1, 2])
    qml.PauliRot(2 * (np.pi - x[1]) * (np.pi - x[2]), 'Z',  wires=2)
    qml.CNOT(wires=[1, 2])
    
    qml.CNOT(wires=[0, 3])
    qml.PauliRot(2 * (np.pi - x[0]) * (np.pi - x[3]), 'Z',  wires=3)
    qml.CNOT(wires=[0, 3])

    qml.CNOT(wires=[1, 3])
    qml.PauliRot(2 * (np.pi - x[1]) * (np.pi - x[3]), 'Z',  wires=3)
    qml.CNOT(wires=[1, 3])
        

@qml.qnode(dev)
def large_circuit(theta, x):
    statepreparation(x)
    qml.RZ(theta[0], wires=0)
    qml.RZ(theta[1], wires=1)
    qml.RZ(theta[2], wires=2)
    qml.RZ(theta[3], wires=3)

    qml.RY(theta[4], wires=0)
    qml.RY(theta[5], wires=1)
    qml.RY(theta[6], wires=2)
    qml.RY(theta[7], wires=3)

    qml.CZ(wires=[0, 3])
    qml.CZ(wires=[0, 1])
    qml.CZ(wires=[1, 2])
    qml.CZ(wires=[2, 3])

    qml.RZ(theta[8], wires=0)
    qml.RZ(theta[9], wires=1)
    qml.RZ(theta[10], wires=2)
    qml.RZ(theta[11], wires=3)

    qml.RY(theta[12], wires=0)
    qml.RY(theta[13], wires=1)
    qml.RY(theta[14], wires=2)
    qml.RY(theta[15], wires=3)
    return qml.expval(qml.PauliZ(0))


@qml.cut_circuit()
@qml.qnode(dev)
def small_circuit(theta, x):
    statepreparation(x)
    qml.RZ(theta[0], wires=0)
    qml.RZ(theta[1], wires=1)
    qml.RZ(theta[2], wires=2)
    qml.RZ(theta[3], wires=3)

    qml.RY(theta[4], wires=0)
    qml.RY(theta[5], wires=1)
    qml.RY(theta[6], wires=2)
    qml.RY(theta[7], wires=3)

    qml.CZ(wires=[0, 3])
    qml.CZ(wires=[0, 1])
    qml.WireCut(wires=1)
    qml.WireCut(wires=2)
    qml.WireCut(wires=3)
    qml.CZ(wires=[1, 2])
    qml.CZ(wires=[2, 3])

    qml.RZ(theta[8], wires=0)
    qml.RZ(theta[9], wires=1)
    qml.RZ(theta[10], wires=2)
    qml.RZ(theta[11], wires=3)

    qml.RY(theta[12], wires=0)
    qml.RY(theta[13], wires=1)
    qml.RY(theta[14], wires=2)
    qml.RY(theta[15], wires=3)
    return qml.expval(qml.PauliZ(0))

data = digits(method='svd', n_components=4)
cv_lst = cross_validation_split(data, n_folds=5)
test = cv_lst[0]
weights = np.array(np.random.random(16), requires_grad=True)
print(small_circuit(weights, test['train']['X'][0, :]))


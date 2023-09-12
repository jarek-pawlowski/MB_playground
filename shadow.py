import numpy as np
import functools as ft

from qiskit import QuantumCircuit, execute, transpile
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from qiskit.extensions import UnitaryGate
from qiskit import Aer

import utils


def measure_in_a_given_basis(psi, basis, backend):
    circ = QuantumCircuit(psi.num_qubits)
    circ.initialize(psi)
    # apply Clifford gates 
    for iq in range(psi.num_qubits):
        circ.append(UnitaryGate(utils.measure_basis_transf[basis[iq]]), [iq])
    # add measurement to the circuit
    meas = QuantumCircuit(psi.num_qubits, psi.num_qubits)
    meas.barrier(range(psi.num_qubits))
    meas.measure(range(psi.num_qubits), range(psi.num_qubits))
    qc = meas.compose(circ, range(psi.num_qubits), front=True)
    qc_compiled = transpile(qc, backend)
    return backend.run(qc_compiled, shots=1, memory=True).result().get_memory()


# start the experiment
no_qubits = 3
# create circuit generating n-qubit GHZ state
circ = QuantumCircuit(no_qubits)
circ.h(0)
for iq in range(1,circ.width()):
    circ.cx(0, iq)

# calculate initial state
psi0 = Statevector.from_int(0, 2**3)
psi0 = psi0.evolve(circ)
rho0 = DensityMatrix(psi0)

# Run the quantum circuit on a statevector simulator backend
backend_measure = Aer.get_backend('qasm_simulator')

# get T snapshots to create shadow
T = 1000
snapshots = []
for it in range(T):
    basis = np.random.randint(3, size=no_qubits)
    measurement = measure_in_a_given_basis(psi0.copy(), basis, backend_measure)
    snapshots.append([basis, [1-int(i)*2 for i in measurement[0]][::-1]])  # measurements are returned in reversed order!
    
# now try to reconstruct the state
rho1 = np.zeros_like(rho0.data)
for it in range(T):
    basis, measurement = snapshots[it]
    single_rho_collection = []
    for iq in range(rho0.num_qubits):
        single_rho_collection.append((np.eye(2)+measurement[iq]*utils.pauli_vector[basis[iq]]*3)/2.)  # (Eq. A5) in https://arxiv.org/pdf/2106.12627.pdf
    rho1 += ft.reduce(np.kron, single_rho_collection)  # Kronecker product of (multiple) single-qubit rhos 
rho1 /= T
    
np.set_printoptions(linewidth=200)
print(np.around(rho0.data, 3))
print(np.around(rho1, 3))

# Pomiar robic po kolei, za kazdym razem dany pomiar kolapsuje ten spin (nalezy zaaplikowac odpowiedni operator rzutowy).
# Wartosc oczekiwana takiego operatora rzutowego (wybrana baza dla wybranego kubitu) daje p-stwo danego wyniku pomiaru. 
# Nastepnie sprawdzamy losujac z przedzialu [0,1] czy taki pomiar sie zdarzyl: wylosowana liczba < p-stwa. 
# Jesli tak, zostawiamy, jesli nie, to nalezy zaaplikowac komplemetarny operator rzutowy.
# Nastepnie przechodzimy do kolejnego kubitu itd.
# https://www2.seas.gwu.edu/~simhaweb/quantum/modules/module5/module5.html
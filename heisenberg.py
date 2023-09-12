import numpy as np

import tenpy

from tenpy.algorithms import dmrg
from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS

import utils

tenpy.tools.misc.setup_logging(to_stdout="INFO")
np.set_printoptions(precision=5, suppress=True, linewidth=100)


# Let's define some spin-chain  
model = XXZChain(dict(L=3, Jxx=1., Jz=1., hz=0.))
L = model.lat.Ls[0]

psi = MPS.from_product_state(model.lat.mps_sites(), (["up", "down"] * L)[:L], model.lat.bc_MPS, dtype=complex)
Szs = psi.expectation_value('Sz')
print("<Sz> =", Szs)
dmrg_params = {"trunc_params": {"chi_max": 100, "svd_min": 1.e-10}, "mixer": True}
info = dmrg.run(psi, model, dmrg_params)
print("E =", info['E'])
Szs = psi.expectation_value(['Sz']*psi.L)
print("<Sz> =", Szs)

# Now start the experiment with the shadow
psi0 = psi
rho0 = psi0.get_rho_segment(range(psi0.L)).to_ndarray().reshape(2**psi0.L,2**psi0.L)

breakpoint()

def measure_in_a_given_basis(psi, basis):
    # apply Clifford gates 
    ops = [utils.op_from_ndarray(utils.measure_basis_transf[basis[iq]]) for iq in range(psi.L)]
    psi.apply_product_op(ops, unitary=True, renormalize=True)
    # add measurement to the circuit
    return psi.sample_measurements(norm_tol=1.)  # ???
    
# get T snapshots to create shadow
T = 1000
snapshots = []
for it in range(T):
    basis = np.random.randint(3, size=psi0.L)
    measurement = measure_in_a_given_basis(psi0.copy(), basis)
    snapshots.append([basis, [1-int(i)*2 for i in measurement[0]][::-1]])
    
# now try to reconstruct the state
import functools as ft
rho1 = np.zeros_like(rho0)
for it in range(T):
    basis, measurement = snapshots[it]
    single_rho_collection = []
    for iq in range(psi.L):
        single_rho_collection.append((np.eye(2)+measurement[iq]*utils.pauli_vector[basis[iq]]*3)/2.)  # (Eq. A5) in https://arxiv.org/pdf/2106.12627.pdf
    rho1 += ft.reduce(np.kron, single_rho_collection)  # Kronecker product of (multiple) single-qubit rhos 
rho1 /= T
    
np.set_printoptions(linewidth=200)
print(np.around(rho0, 3))
print(np.around(rho1, 3))
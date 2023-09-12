import numpy as np

from tenpy.tools.params import asConfig
from tenpy.networks.site import SpinHalfSite 
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
import tenpy.linalg.np_conserved as npc


# define basis transformations (Clifford gates), 
# see: https://quantumcomputing.stackexchange.com/questions/13605/how-to-measure-in-another-basis
hadamard = np.array([[ 1. , 1. ],[ 1. ,-1. ]])/np.sqrt(2.)
s_dag    = np.array([[ 1. , 0. ],[ 0. ,-1.j]])
measure_basis_transf = [hadamard, hadamard @ s_dag, np.eye(2)]

# Pauli vector
sx = np.array([[ 0. , 1. ],[ 1. , 0. ]])
sy = np.array([[ 0. ,-1.j],[ 1.j, 0. ]])
sz = np.array([[ 1. , 0. ],[ 0. ,-1. ]])
pauli_vector = [sx, sy, sz]

# create a ChargeInfo to specify the nature of the charge
chinfo = npc.ChargeInfo([1], ['2*Sz'])  # the second argument is just a descriptive name
# create LegCharges on physical leg
p_leg = npc.LegCharge.from_qflat(chinfo, [[1], [-1]])  # charges for up, down

def op_from_ndarray(array):
    return npc.Array.from_ndarray(array, [p_leg, p_leg.conj()], labels=['p', 'p*'], raise_wrong_sector=False)


class XXZChain1(CouplingModel, NearestNeighborModel, MPOModel):
    r"""Spin-1/2 XXZ chain with Sz conservation.

    The Hamiltonian reads:

    .. math ::
        H = \sum_i \mathtt{Jxx}/2 (S^{+}_i S^{-}_{i+1} + S^{-}_i S^{+}_{i+1})
                 + \mathtt{Jz} S^z_i S^z_{i+1} \\
            - \sum_i \mathtt{hz} S^z_i

    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`XXZChain` below.

    Options
    -------
    .. cfg:config :: XXZChain
        :include: CouplingMPOModel

        L : int
            Length of the chain.
        Jxx, Jz, hz : float | array
            Coupling as defined for the Hamiltonian above.
        bc_MPS : {'finite' | 'infinte'}
            MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.

    """
    def __init__(self, model_params):
        # 0) read out/set default parameters
        model_params = asConfig(model_params, "XXZChain")
        L = model_params.get('L', 2)
        Jxx = model_params.get('Jxx', 1.)
        Jz = model_params.get('Jz', 1.)
        hz = model_params.get('hz', 0.)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        sort_charge = model_params.get('sort_charge', None)
        # 1-3):
        site = SpinHalfSite(conserve='Sz', sort_charge=sort_charge)
        # 4) lattice
        bc = 'open' if bc_MPS == 'finite' else 'periodic'
        lat = Chain(L, site, bc=bc, bc_MPS=bc_MPS)
        # 5) initialize CouplingModel
        CouplingModel.__init__(self, lat)
        # 6) add terms of the Hamiltonian
        # (u is always 0 as we have only one site in the unit cell)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hz, u, 'Sz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(Jxx * 0.5, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)
        # 7) initialize H_MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        # 8) initialize H_bond (the order of 7/8 doesn't matter)
        NearestNeighborModel.__init__(self, lat, self.calc_H_bond())
      
      
class XXZChain2(CouplingMPOModel, NearestNeighborModel):
    """Another implementation of the Spin-1/2 XXZ chain with Sz conservation.

    This implementation takes the same parameters as the :class:`XXZChain`, but is implemented
    based on the :class:`~tenpy.models.model.CouplingMPOModel`.

    Parameters
    ----------
    model_params : dict | :class:`~tenpy.tools.params.Config`
        See :cfg:config:`XXZChain`
    """
    default_lattice = "Chain"
    force_default_lattice = True

    def init_sites(self, model_params):
        sort_charge = model_params.get('sort_charge', None)
        return SpinHalfSite(conserve='Sz', sort_charge=sort_charge)  # use predefined Site

    def init_terms(self, model_params):
        # read out parameters
        Jxx = model_params.get('Jxx', 1.)
        Jz = model_params.get('Jz', 1.)
        hz = model_params.get('hz', 0.)
        # add terms
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hz, u, 'Sz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(Jxx * 0.5, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx) 

from src.kalman.base import LGSSM
from src.kernels.base import ContinuousDiscreteModel


def sde_to_lgssm(sde: ContinuousDiscreteModel) -> LGSSM:

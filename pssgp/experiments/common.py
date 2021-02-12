import enum

from absl import flags
from gpflow.kernels import SquaredExponential
from gpflow.models import GPR

from pssgp.kernels import Matern12, Matern32, Matern52, RBF, Periodic
from pssgp.model import StateSpaceGP


class ModelEnum(enum.Enum):
    GP = "GP"
    SSGP = "SSGP"
    PSSGP = "PSSGP"


class CovarianceEnum(enum.Enum):
    Matern12 = 'Matern12'
    Matern32 = 'Matern32'
    Matern52 = 'Matern52'
    RBF = "RBF"
    QP = "QP"


flags.DEFINE_string("device", "/cpu:0", "Device on which to run")

def get_covariance_function(covariance_enum, **kwargs):
    if covariance_enum == CovarianceEnum.Matern12:
        return Matern12(**kwargs)
    if covariance_enum == CovarianceEnum.Matern32:
        return Matern32(**kwargs)
    if covariance_enum == CovarianceEnum.Matern52:
        return Matern52(**kwargs)
    if covariance_enum == CovarianceEnum.RBF:
        return RBF(**kwargs)
    if covariance_enum == CovarianceEnum.QP:
        base_kernel = SquaredExponential(kwargs.pop("variance"), kwargs.pop("lengthscales"))
        return Periodic(base_kernel, **kwargs)


def get_model(model_enum, data, noise_variance, covariance_function, max_parallel=10000):
    if model_enum == ModelEnum.GP:
        return GPR(data, covariance_function, None, noise_variance)
    if model_enum == ModelEnum.SSGP:
        return StateSpaceGP(data, covariance_function, noise_variance, parallel=False)
    if model_enum == ModelEnum.PSSGP:
        return StateSpaceGP(data, covariance_function, noise_variance, parallel=True, max_parallel=max_parallel)

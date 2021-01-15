import gpflow
from gpflow import Parameter
from gpflow.kernels import Kernel
from gpflow.models import GPModel
from gpflow.models.model import MeanAndVariance
from gpflow.models.training_mixins import InputData, RegressionData
from gpflow.models.util import data_input_to_tensor


class StateSpaceGP(GPModel):
    def __init__(self,
                 data: RegressionData,
                 kernel: Kernel,
                 noise_variance: float = 1.0,
                 ):
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        self._noise_variance = Parameter(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, None, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)

    def _make_model(self, ts, ):

        def predict_f(
                self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
        ) -> MeanAndVariance:

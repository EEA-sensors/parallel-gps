from gpflow.kernels import Kernel
from gpflow.models import GPModel
from gpflow.models.model import MeanAndVariance
from gpflow.models.training_mixins import InputData, RegressionData

def _discretize_sde(sde_model, )


class StateSpaceGP(GPModel):
    def __init__(self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: float = 1.0,
    ):
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)
    def _make_model(self, ts, ):
    def predict_f(
            self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:


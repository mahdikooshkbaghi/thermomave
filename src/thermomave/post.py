""" postprocess utilities"""
import jax.numpy as jnp
from numpyro.infer import Predictive
from jax.numpy import DeviceArray


class ppc:
    """Posterior or prior predictive"""

    def __init__(self, model=None, method=None, guide=None,
                 svi_params=None, trace=None,
                 x: DeviceArray = None, rng_key=None, num_ppc: int = 100):

        assert model != None, f"For ppc, model instance needs to pass as an argument"
        self.model = model
        assert method != None, f"For ppc, method=svi or method=mcmc needs to pass as an argument"
        self.method = method
        if self.method == 'svi':
            assert guide != None, f"For ppc of SVI, guide needs to pass as an argument"
            self.guide = guide
            assert svi_params != None, f"For ppc of SVI, svi_params needs to pass as an argument"
            self.svi_params = svi_params
        if self.method == 'mcmc':
            assert trace != None, f"For ppc of mcmc, trace needs to pass as an argument"
            self.trace = trace
        assert x != None, f"For ppc, the input features are needed to pass as an argument"
        self.x = x
        self.rng_key = rng_key
        self.num_ppc = num_ppc
        print(self.num_ppc)

    def summary(self, samples):
        site_stats = {}
        q = [2.5, 97.5]
        for k, v in samples.items():
            site_stats[k] = {
                "mean": jnp.mean(v, axis=0),
                "std": jnp.std(v, 0),
                "q": jnp.percentile(v, q, axis=0)
            }
        return site_stats

    def pred(self):

        if self.method == 'svi':
            predictive = Predictive(model=self.model, guide=self.guide,
                                    params=self.svi_params, num_samples=self.num_ppc)
            samples = predictive(self.rng_key, x=self.x)
            self.model_predictions = self.summary(samples)

            self.parameters_predictions = self.guide.sample_posterior(self.rng_key, self.svi_result.params,
                                                                      sample_shape=(self.num_ppc,))

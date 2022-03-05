# standard import
from typing import Optional
import time
# numpyro imports
from numpyro.infer import SVI, autoguide
from numpyro.infer import init_to_sample
from numpyro.infer import Trace_ELBO
import numpyro.optim as optim
from numpyro.infer import NUTS, MCMC
# jax imports
from jax.numpy import DeviceArray
import jax.numpy as jnp


class fit():
    def __init__(self,
                 args,
                 rng_key,
                 model: DeviceArray = None):

        self.rng_key = rng_key
        self.model = model
        self.num_samples = args.num_samples
        self.num_warmup = args.num_warmup
        self.batch_size = args.batch_size
        self.num_chains = args.num_chains
        self.step_size = args.step_size
        self.learning_decay = args.learning_decay

    def svi(self, x, y):
        print('\nTraining using Stochastic Variational Inference\n')
        # helper function for running SVI with a particular autoguide
        start = time.time()
        guide = autoguide.AutoDelta(
            self.model, init_loc_fn=init_to_sample)

        # Initial learning rate
        step_size = self.step_size
        init_lr = self.step_size
        if self.learning_decay is not None:
            def step_size(i):
                return init_lr * self.learning_decay**jnp.floor(i / 1_000)
        optimizer = optim.RMSProp(step_size=step_size)
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())

        svi_result = svi.run(
            rng_key=self.rng_key, num_steps=self.num_samples, x=x, y=y, batch_size=self.batch_size)
        print("\nVariational inference elapsed time:", time.time() - start)
        return guide, svi_result

    def mcmc(self, x, y):
        print('\nTraining using MCMC\n')
        start = time.time()
        # define kernel
        kernel = NUTS(model=self.model)
        # setup mcmc
        mcmc = MCMC(kernel,
                    num_warmup=self.num_warmup,
                    num_samples=self.num_samples,
                    num_chains=self.num_chains)
        # run mcmc inference
        mcmc.run(self.rng_key, x=x, y=y)
        print("\nMCMC elapsed time:", time.time() - start)
        return mcmc

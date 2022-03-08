# standard import
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
        self.device = args.device

        error_samp = f"number of samples for mcmc/svi inference should be provided as args"
        # Assign number of samples for inference.
        assert args.num_samples != None, error_samp
        self.num_samples = args.num_samples

        # Assign parameters for the MCMC inference
        if args.method == 'mcmc':
            error_warm = f"number of warmup steps for mcmc inference should be provided as args"
            error_chain = f"number of chains for mcmc inference should be provided as args"

            # Assign number of warmup steps for MCMC inference.
            assert args.num_warmup != None, error_warm
            self.num_warmup = args.num_warmup

            # Assign number of chains for MCMC inference.
            assert args.num_chains != None, error_chain
            self.num_chains = args.num_chains

        # Assign parameters for the SVI inference.
        if args.method == 'svi':
            # Assign the learning rate
            learning_rate = getattr(args, 'learning_rate', None)
            if learning_rate == None:
                self.step_size = 1e-1
                print('\nDefault Learning rate = 1e-1 is used')
            else:
                self.step_size = learning_rate
            # Assign the batch size
            batch_size = getattr(args, 'batch_size', None)
            if batch_size == None:
                self.batch_size = 128
                print('\nDefault batch size = 128 is used')
            else:
                self.batch_size = batch_size

            # Assign Learning rate decay
            learning_decay = getattr(args, 'learning_decay', None)
            if learning_decay == None:
                self.learning_decay = None
            else:
                self.learning_decay = learning_decay

    def svi(self, x: DeviceArray, y: DeviceArray):
        """
        Stochastic Variational Inference.

        Parameters
        ----------
        x: (DeviceArray):
            Input features for training.
        y: (DeviceArray):
            Output (measurements) for training.

        Returns
        -------
        guide: (numpyro.infer.autoguide.AutoDelta):
            Numpyro Automatic Guide Generated.

        svi_results: (SVIRunResult)
            Stochastic Variational Inference Result.
        """

        print('\nTraining using Stochastic Variational Inference\n')
        start = time.time()

        # Assign Autodelta Autoguide.
        guide = autoguide.AutoDelta(
            self.model, init_loc_fn=init_to_sample)

        # Initial learning rate
        step_size = self.step_size
        init_lr = self.step_size

        # Reduce learning rate based on the rate in learning decay for
        # every 1000 epochs
        if self.learning_decay is not None:
            def step_size(i):
                return init_lr * self.learning_decay**jnp.floor(i / 1_000)

        # Define the optimizer
        optimizer = optim.RMSProp(step_size=step_size)

        # Loss function is the ELBO
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())

        svi_results = svi.run(
            rng_key=self.rng_key, num_steps=self.num_samples,
            x=x, y=y, batch_size=self.batch_size)

        print("\nVariational inference elapsed time:", time.time() - start)
        return guide, svi_results

    def mcmc(self, x, y):
        print('\nTraining using MCMC\n')
        start = time.time()
        # define kernel
        kernel = NUTS(model=self.model)
        # setup mcmc
        mcmc = MCMC(kernel,
                    num_warmup=self.num_warmup,
                    num_samples=self.num_samples,
                    num_chains=self.num_chains,
                    progress_bar=False if self.device is 'gpu' else True)
        # run mcmc inference
        mcmc.run(self.rng_key, x=x, y=y)
        print("\nMCMC elapsed time:", time.time() - start)
        return mcmc

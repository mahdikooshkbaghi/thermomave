# dill for saving model
from typing import Optional
import dill
# jax import
import jax.random as random
import jax.numpy as jnp
from jax.numpy import DeviceArray
# numpyro imports
import numpyro
import numpyro.distributions as dist
from . infer import fit


class ModelHandler:
    """
    Represents a numpyro thermodynamic model handler.

    Parameters
    ----------
    L: (int)
        Length of each training sequence. Must be ``>= 1``.

    C: (int)
        Length of the alphabet in the sequence.

    D_H: (int)
        Number of nodes in the nonlinearity maps latent phenotype to
        measurements. Default = 20.

    kT: (float)
        Boltzmann constant. Default = 0.582 (room temperature).

    ge_noise_model_type: (str)
            Specifies the type of noise model the user wants to infer.
            The possible choices allowed: ['Gaussian','Cauchy','SkewedT', 'Empirical']

    """

    def __init__(self, L: int, C: int, D_H: int = 20, kT: float = 0.582,
                 ge_noise_model_type: str = 'Gaussian'):
        self.L = L
        self.D_H = D_H
        self.C = C
        self.kT = kT
        self.ge_noise_model_type = ge_noise_model_type

    def nonlin(self, x):
        """
        Define nonlinear function mapping the latent phenotype to measurements.
        """
        return jnp.tanh(x)

    def model(self, x: DeviceArray = None, y: DeviceArray = None,
              batch_size: int = None) -> DeviceArray:
        """
        Numpyro model instance.
        """
        L = self.L
        C = self.C
        D_H = self.D_H
        kT = self.kT

        # Initialize constant parameter for folding energy
        theta_f_0 = numpyro.sample(
            "theta_f_0", dist.Normal(loc=0, scale=1))

        # Initialize constant parameter for binding energy
        theta_b_0 = numpyro.sample(
            "theta_b_0", dist.Normal(loc=0, scale=1))

        # Initialize additive parameter for folding energy
        theta_f_lc = numpyro.sample(
            "theta_f_lc", dist.Normal(loc=jnp.zeros((L, C)),
                                      scale=jnp.ones((L, C))))

        # Initialize additive parameter for binding energy
        theta_b_lc = numpyro.sample(
            "theta_b_lc", dist.Normal(loc=jnp.zeros((L, C)),
                                      scale=jnp.ones((L, C))))

        # Compute Delta G for binding
        Delta_G_f = numpyro.deterministic(
            "Delta_G_f", theta_f_0 + jnp.einsum('ij,kij->k', theta_f_lc, x))
        Delta_G_f = Delta_G_f[..., jnp.newaxis]
        # Compute Delta G for folding
        Delta_G_b = numpyro.deterministic(
            "Delta_G_b", theta_b_0 + jnp.einsum('ij,kij->k', theta_b_lc, x))
        Delta_G_b = Delta_G_b[..., jnp.newaxis]

        # Compute and return fraction folded and bound
        Z = numpyro.deterministic(
            'z', 1 + jnp.exp(-Delta_G_f / kT) + jnp.exp(-(Delta_G_f + Delta_G_b) / kT))
        if y is not None:
            assert Z.shape == y.shape, f"Z has shape {Delta_G_b.shape}, y has shape {y.shape}"

        # Latent phenotype
        phi = numpyro.deterministic("phi",
                                    (jnp.exp(-(Delta_G_f + Delta_G_b) / kT)) / Z)
        if y is not None:
            assert phi.shape == y.shape, f"phi has shape {phi.shape}, y has shape {y.shape}"

        # GE parameters
        a = numpyro.sample("a", dist.Normal(loc=0, scale=1))
        b = numpyro.sample("b", dist.Normal(
            jnp.zeros((D_H, 1)), jnp.ones((D_H, 1))))
        c = numpyro.sample("c", dist.Normal(
            jnp.zeros((D_H, 1)), jnp.ones((D_H, 1))))
        d = numpyro.sample("d", dist.Normal(
            jnp.zeros((D_H, )), jnp.ones((D_H, ))))

        # GE regression
        tmp = jnp.einsum('ij, kj->ki', c, phi)
        g = numpyro.deterministic(
            "g", a + jnp.einsum('ij, ki->kj', b, self.nonlin(tmp + d[None, :])))
        if y is not None:
            assert g.shape == y.shape, f"g has shape {g.shape}, y has shape {y.shape}"

        # noise = numpyro.sample("noise", dist.Gamma(3.0, 1.0))
        self.alpha, self.beta, noise = self.noise_model(
            self.ge_noise_model_type)
        sigma_obs = 1.0 / jnp.sqrt(noise)
        with numpyro.plate("data", x.shape[0], subsample_size=batch_size) as ind:
            if y is not None:
                batch_y = y[ind]
            else:
                batch_y = None
            batch_g = g[ind]
            return numpyro.sample("yhat", dist.Normal(
                batch_g, sigma_obs).to_event(1), obs=batch_y)

    def noise_model(self, ge_noise_model_type):
        """
        Define the Global Epistasis Noise Model.
        """

        # Check the ge_noise_model_type is in the list of implemented one.
        error_mess = f"ge_noise_model_type should be 'Gamma' "
        assert ge_noise_model_type in ['Gamma'], error_mess

        # Gamma noise model
        if ge_noise_model_type == 'Gamma':
            alpha = numpyro.sample('alpha', dist.Uniform(0.5, 5))
            beta = numpyro.sample('beta', dist.Uniform(0.5, 2))
            return alpha, beta, numpyro.sample("noise", dist.Gamma(alpha, beta))
    # use the fit class as the ModelHandler instance

    def fit(self, args, x: DeviceArray, y: DeviceArray, rng_key: Optional[int] = None):
        if rng_key is None:
            rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        self.method = args.method
        if args.method == 'svi':
            self.guide, self.svi_results = fit(
                args=args, rng_key=rng_key, model=self.model).svi(x=x, y=y)
            return self.guide, self.svi_results
        if args.method == 'mcmc':
            self.trace = fit(args=args, rng_key=rng_key,
                             model=self.model).mcmc(x=x, y=y)
            return self.trace

    def save(self, filepath=None):
        # This is not working need change.
        self.filepath = filepath
        print(f'Saving model to {self.filepath}')
        output_dict = {}
        if self.method == 'svi':
            # output_dict['model'] = self.model
            output_dict['guide'] = self.guide
            output_dict['svi_params'] = self.svi_results.params
            with open(self.filepath, 'wb') as handle:
                dill.dump(output_dict, handle)
        if self.method == 'mcmc':
            output_dict['model'] = self.model
            output_dict['trace'] = self.trace
            with open(self.filepath, 'wb') as handle:
                dill.dump(output_dict, handle)

from typing import Optional
# dill for saving model
import dill
import pandas as pd
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

    The construction of the model is based on two dataframes: sate_df and energy_df.

    state_df: 
    Must includes the 'States' and 'Activity' columns. It should include the 
    name of the energy as we want to have in the model and the corresponding coefficients for
    each state. These coefficients define the functional form of the energy as a linear combination
    between them.
    Example (state_df): Here is the state_df for sorsteq model.

    | State    | Activity | G_R | G_C | G_I |
    | -------- | -------- | --- | --- | --- |
    | Empty    | 0        | 0   | 0   | 0   |
    | CRP      | 0        | 0   | 1   | 0   |
    | RNAP     | 1        | 1   | 0   | 0   |
    | RNAP+CRP | 1        | 1   | 1   | 1   |

    energy_df: 
    This dataframe provide the `Type` of energy as well as 
    starting `Start` and stopping `Stop` location that the molecule for that 
    specific energy is binding.

    Example (energy_df): Here is the state_df for sorsteq model.
    G_C: CRP energy.
    G_R: RNAP energy.
    G_I: Interaction energy.

    | Energies     | Type     | Start | Stop |
    | ------------ | -------- | ----- | ---- |
    | G_C          | additive | 1     | 27   |
    | G_R          | additive | 34    | 75   |
    | G_I          | scalar   | NA    | NA   |

    Parameters
    ----------
    L: (int)
        Length of each training sequence. Must be ``>= 1``.

    C: (int)
        Length of the alphabet in the sequence.

    state_df: (pd.Dataframe)
            The state dataframe.

    energy_df: (pd.Dataframe)
            The energy dataframe.

    D_H: (int)
        Number of nodes in the nonlinearity maps latent phenotype to
        measurements. Default = 20.

    kT: (float)
        Boltzmann constant. Default = 0.582 (room temperature).

    ge_noise_model_type: (str)
            Specifies the type of noise model the user wants to infer.
            The possible choices allowed: ['Gaussian','Cauchy','SkewedT', 'Empirical']

    """

    def __init__(self, L: int, C: int,
                 state_df: pd.Dataframe,
                 energy_df: pd.Dataframe,
                 D_H: int = 20, kT: float = 0.582,
                 ge_noise_model_type: str = 'Gaussian'):

        # Assing the sequence length.
        self.L = L
        # Assing the alphabet length.
        self.C = C
        # Assign the state_df to the layer.
        self.state_df = state_df
        # Assign the energy_df to the layer.
        self.energy_df = energy_df
        # Assign the number of hidden nodes in the GE regression.
        self.D_H = D_H
        # Assign the Boltzmann constant.
        self.kT = kT
        # Assign the ge noise model type.
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

        # Get list of energy names
        energy_list = self.energy_df['Energies'].values

        for eng_name in energy_list:
            # Find the corresponding row in the energy_df
            ix = self.energy_df[self.energy_df['Energies'] == eng_name]
            # Find the type of energy to assign the theta.
            # There are two options 1. additive 2. scalar.
            eng_type = ix['Type'].values

            # Additive parameters
            if eng_type == 'additive':
                # Find the starting position on the sequence
                start_idx = int(ix['Start'].values)
                # Find the stopping position on the sequence
                stop_idx = int(ix['Stop'].values)
                # Find the length of interest on the sequence
                l_lc = int(ix['l_lc'].values)
                # Find the shape of theta_lc
                theta_lc_shape = int(ix['l_lc'].values)

                # Create the theta_0 name: insert theta_0 to the begining of the energy names
                theta_0_name = f'theta_0_{eng_name}'
                # Prior on the theta_0
                theta_0 = numpyro.sample(
                    theta_0_name, dist.Normal(loc=0, scale=1))
                # Create the theta_lc name: insert theta_lc to the begining of the energy names
                theta_lc_name = f'theta_lc_{eng_name}'
                # Prior on the theta_lc
                theta_lc = numpyro.sample(theta_lc_name, dist.Normal(loc=jnp.zeros((theta_lc_shape, C)),
                                                                     scale=jnp.ones((theta_lc_shape, C))))

                x_eng = x[:, C * start_idx:C * stop_idx]
                x_lc = jnp.reshape(x_eng, [-1, l_lc, C])
                Delta_G_name = f'Delta_G_{eng_name}'
                Delta_G = numpyro.deterministic(
                    Delta_G_name, theta_0 + jnp.einsum('ij,kij->k', theta_lc, x_lc))
                Delta_G = Delta_G[..., jnp.newaxis]

            # Scalar parameters
            if eng_type == 'scalar':
                theta_0_name = f'theta_0{eng_name}'
                # Prior on the theta_0
                theta_0 = numpyro.sample(
                    theta_0_name, dist.Normal(loc=0, scale=1))

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

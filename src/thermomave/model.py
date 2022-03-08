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

    state_df: (pd.DataFrame)
            The state dataframe.

    energy_df: (pd.DataFrame)
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
                 state_df: pd.DataFrame,
                 energy_df: pd.DataFrame,
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
        # Make new column in the energy_df for stop-stop length
        self.energy_df['l_lc'] = self.energy_df['Stop'] - \
            self.energy_df['Start']
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
        energy_df = self.energy_df
        state_df = self.state_df

        # Get list of energy names
        energy_list = energy_df['Energies'].values

        # Initialize the theta dictionary
        theta_dict = {}

        # Create priors dictionary for theta
        for eng_name in energy_list:
            # Find the corresponding row in the energy_df
            ix = energy_df[energy_df['Energies'] == eng_name]
            # Find the type of energy to assign the theta.
            # There are two options 1. additive 2. scalar.
            eng_type = ix['Type'].values

            # Additive parameters
            if eng_type == 'additive':
                # Create the theta_0 name: insert theta_0 to the begining of the energy names
                theta_0_name = f'theta_0_{eng_name}'
                # Prior on the theta_0
                theta_dict[theta_0_name] = numpyro.sample(
                    theta_0_name, dist.Normal(loc=0, scale=1))
                # Create the theta_lc name: insert theta_lc to the begining of the energy names
                theta_lc_name = f'theta_lc_{eng_name}'
                # Find the shape of theta_lc
                theta_lc_shape = int(ix['l_lc'].values)
                # Prior on the theta_lc
                theta_dict[theta_lc_name] = numpyro.sample(theta_lc_name, dist.Normal(loc=jnp.zeros((theta_lc_shape, C)),
                                                                                      scale=jnp.ones((theta_lc_shape, C))))
            # Scalar parameters
            if eng_type == 'scalar':
                theta_0_name = f'theta_0{eng_name}'
                # Prior on the theta_0
                theta_dict[theta_0_name] = numpyro.sample(
                    theta_0_name, dist.Normal(loc=0, scale=1))

        # Initialize the energy dictionary
        G_dict = {}

        for eng_name in energy_list:
            # Find the corresponding row in the energy_df
            ix = self.energy_df[self.energy_df['Energies'] == eng_name]
            # Find the type of energy which used to assign the theta dict.
            # There are two options 1. additive 2. scalar
            eng_type = ix['Type'].values

            # If there is additive G, reterive theta and calculate G values
            if eng_type == 'additive':
                # Get the name of theta_0 parameters
                theta_0_name = f'theta_0_{eng_name}'
                # Get the name of theta_lc parameters
                theta_lc_name = f'theta_lc_{eng_name}'
                # Find the starting position on the sequence
                start_idx = int(ix['Start'].values)
                # Find the stopping position on the sequence
                stop_idx = int(ix['Stop'].values)
                # Find the length of interest on the sequence
                l_lc = int(ix['l_lc'].values)
                # Reshape input to samples x length x characters
                x_eng = x[:, C * start_idx:C * stop_idx]
                x_lc = jnp.reshape(x_eng, [-1, l_lc, C])
                # Compute Delta G
                G_dict[eng_name] = numpyro.deterministic(f'G_{eng_name}',
                                                         theta_dict[theta_0_name] + jnp.einsum('ij,kij->k', theta_dict[theta_lc_name], x_lc))

            # If the energy type is scalar, the G=theta
            if eng_type == 'scalar':
                theta_0_name = f'theta_0_{eng_name}'
                G_dict[eng_name] = theta_dict[theta_0_name]

        # Compute partition function
        # Z: total partion function
        Z = 0.0
        # prob: total probability
        prob = 0.0

        # Loop over States in state_df
        for state in state_df['States']:
            # Compute total energy for each state
            total_eng_state = 0.0

            for eng_name in energy_list:
                # Find the index of energy in the state_df
                jx = state_df[state_df['States'] == state]
                # get the coefficient of the energy for each state from the state_df
                coef = float(jx[eng_name].values)
                # For each state partial partion function values is coef*G_dict[eng_name]
                total_eng_state = total_eng_state + coef * G_dict[eng_name]
            # parital partion function for each state
            Z_state = jnp.exp(-total_eng_state / kT)

            # Get the activity of each state
            activity = float(jx['Activity'].values)
            # probability of the state
            prob_state = activity * Z_state

            # Update total partition function
            Z = Z + Z_state
            # Update total probability
            prob = prob + prob_state

        # Return the latent variable
        phi = numpyro.deterministic('phi', prob / Z)
        phi = phi[..., jnp.newaxis]
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

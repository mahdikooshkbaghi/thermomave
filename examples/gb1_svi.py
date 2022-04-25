from dataclasses import dataclass
from turtle import pos
import pandas as pd
import numpyro
import jax.random as random
import dill
import sys
path_to_thermomav_local = '../src'
sys.path.insert(0, path_to_thermomav_local)
from thermomave import utils
from thermomave.model import ModelHandler


@dataclass
class args:
    num_chains = 4
    num_samples = 10_000
    num_warmup = 10
    learning_rate = 1e-3
    batch_size = 300
    device = 'cpu'
    method = 'svi'
    progress_bar = True


numpyro.set_platform(args.device)
numpyro.set_host_device_count(args.num_chains)

# generate random keys for training and predictions
rng_key, rng_key_predict = random.split(random.PRNGKey(0))
# Load dataset
x, y, L, C = utils.load_dataset('../data/gb1.csv.gz')

# Define State Dataframe
state_dict = {
    'States': ['Unfolded', 'Folded', 'Bounded'],
    'Activity': [0, 0, 1],
    'G_f': [0, 1, 1],
    'G_b': [0, 0, 1]
}
state_df = pd.DataFrame(state_dict)


# Define Energy Dataframe
energy_dict = {
    'Energies': ['G_f', 'G_b'],
    'Type': ['additive', 'additive'],
    'Start': [0, 0],
    'Stop': [L, L],
}
energy_df = pd.DataFrame(energy_dict)

# Generate model instance
model_handler = ModelHandler(L=L, C=C,
                             state_df=state_df,
                             energy_df=energy_df,
                             D_H=20, kT=0.582,
                             ge_noise_model_type='Gamma')
model_handler.fit(args, x=x, y=y)

# Posterior Predictive
from numpyro.infer import Predictive
import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp


def summary(samples):
    site_stats = {}
    site_stats = {
        "mean": jnp.mean(samples, axis=0),
        "std": jnp.std(samples, 0),
    }
    return site_stats


num_sample_dist = 40
posterior_predictive = Predictive(model=model_handler.model,
                                  guide=model_handler.guide,
                                  params=model_handler.svi_results.params,
                                  num_samples=num_sample_dist)
posterior_predictions = posterior_predictive(rng_key_predict, x=x)

model_prediction = summary(posterior_predictions['yhat'])
# print(summary(posterior_predictions))

# Posterior plots
fig, axs = plt.subplots(1, 2, figsize=(8, 4 / 1.2))
for i in range(num_sample_dist):
    sns.kdeplot(posterior_predictions['yhat'][i, :, 0],
                alpha=0.1,
                ax=axs[0],
                color=(0.000, 0.255, 0.745, 0.1))

yhat_mean = model_prediction['mean']
Rsq = jnp.corrcoef(y.ravel(), yhat_mean.ravel())[0, 1]**2

axs[1].scatter(y, yhat_mean, s=1, alpha=0.1)
axs[1].set_xlabel('model prediction ($\hat{y}$) mean')
axs[1].set_ylabel('measurement ($y$)')
axs[1].set_title(f'Model performance: $R^2$={Rsq:.3}')

sns.kdeplot(y[:, 0], ax=axs[0], color="darkorange",
            lw=3, label="Data Density")
axs[1].plot(y, y, c='r')
axs[0].legend()
plt.tight_layout()

fig, ax = plt.subplots(1, 1)
ax.plot(model_handler.svi_results.losses)
ax.set_xlabel('epoch')
ax.set_ylabel('ELBO')
plt.tight_layout()


ppc = model_handler.guide.sample_posterior(rng_key, model_handler.svi_results.params,
                                           sample_shape=(num_sample_dist,))
print(ppc['a'])

plt.show()

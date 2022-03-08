from dataclasses import dataclass
import pandas as pd
import numpyro
import jax.random as random
import dill
from thermomave import utils
from thermomave.model import ModelHandler


@dataclass
class args:
    num_chains = 4
    num_samples = 10
    learning_rate = 1e-3
    batch_size = 300
    device = 'cpu'
    method = 'svi'


numpyro.set_platform(args.device)
numpyro.set_host_device_count(args.num_chains)

# generate random keys for training and predictions
rng_key, rng_key_predict = random.split(random.PRNGKey(0))
# Load dataset
x, y, L, C = utils.load_dataset('./data/gb1.csv.gz')

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
# print(model_handler.guide.sample_posterior(
# rng_key, model_handler.svi_results.params))
#

#output_dict = {}
#output_dict['model'] = model_handler.model
#output_dict['guide'] = model_handler.guide
#output_dict['svi_params'] = model_handler.svi_results.params
#output_dict['x'] = x
#output_dict['y'] = y
# with open('trace.pkl', 'wb') as handle:
#    dill.dump(output_dict, handle)
# model_handler.save(filepath='a.pkl')


# Posterior Predictive
from numpyro.infer import Predictive
import seaborn as sns
import matplotlib.pyplot as plt
num_sample_dist = 20
posterior_predictive = Predictive(model=model_handler.model,
                                  guide=model_handler.guide,
                                  params=model_handler.svi_results.params,
                                  num_samples=num_sample_dist)
posterior_predictions = posterior_predictive(rng_key_predict, x=x)


# Posterior plots
fig, axs = plt.subplots(1, 2, figsize=(8, 4 / 1.2))
for i in range(num_sample_dist):
    sns.kdeplot(posterior_predictions['yhat'][i, :, 0],
                alpha=0.1,
                ax=axs[0],
                color=(0.000, 0.255, 0.745, 0.1))
    axs[1].scatter(y, posterior_predictions['yhat']
                   [i, :, 0], c='k', s=1, alpha=0.1)
sns.kdeplot(y[:, 0], ax=axs[0], color="darkorange",
            lw=3, label="Training Data Density")
axs[1].plot(y, y, c='r')
plt.tight_layout()
plt.show()

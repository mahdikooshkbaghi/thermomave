from dataclasses import dataclass
import numpyro
import jax.random as random

from thermomave import utils
from thermomave.model import ModelHandler


@dataclass
class args:
    num_chains = 4
    num_samples = 1000
    learning_rate = 1e-3
    batch_size = 256
    device = 'cpu'
    method = 'svi'


numpyro.set_platform(args.device)
numpyro.set_host_device_count(args.num_chains)

# generate random keys for training and predictions
rng_key, rng_key_predict = random.split(random.PRNGKey(0))
# Load dataset
x, y, L, C = utils.load_dataset('./data/gb1.csv.gz')
# Generate model instance
model_handler = ModelHandler(L=L, C=C, D_H=20, kT=0.582,
                             ge_noise_model_type='Gamma')
model_handler.fit(args, x=x, y=y)
print(model_handler.guide.sample_posterior(
    rng_key, model_handler.svi_results.params))
# model_handler.save(filepath='a.pkl')

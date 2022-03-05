from dataclasses import dataclass
import numpyro
import jax.random as random

from thermomave import utils
from thermomave.model import ModelHandler


@dataclass
class args:
    num_chains = 4
    num_warmup = 10
    num_samples = 10
    batch_size = 128
    step_size = 1e-3
    device = 'cpu'
    method = 'svi'
    learning_decay = 0.1


numpyro.set_platform(args.device)
numpyro.set_host_device_count(args.num_chains)

# generate random keys for training and predictions
rng_key, rng_key_predict = random.split(random.PRNGKey(0))
# Load dataset
x, y, L, C = utils.load_dataset('./data/gb1.csv.gz')
# Generate model instance
model_handler = ModelHandler(L=L, C=C, D_H=20, kT=0.582)
model_handler.fit(args, x=x, y=y)
# model_handler.save(filepath='a.pkl')

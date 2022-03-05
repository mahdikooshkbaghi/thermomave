import numpyro
import jax.random as random

from thermomave import utils
from thermomave.model import model
from thermomave.infer import fit
from thermomave.post import ppc
import dill

# generate random keys for training and predictions
rng_key, rng_key_predict = random.split(random.PRNGKey(0))
# Load dataset
x, y, L, C = utils.load_dataset('./data/gb1.csv.gz')


with open('file.pkl', 'rb') as f:
    input_dict = dill.load(f)
model = input_dict['model']
guide = input_dict['guide']
params = input_dict['svi_params']


ppc(model=model, guide=guide, svi_params=params,
    method='svi', x=x, num_ppc=10).pred()

model_pred = ppc.model_predictions()

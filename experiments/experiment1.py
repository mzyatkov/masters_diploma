import pandas as pd
import pymc as pm
from models.bayesian_model import create_bayesian_model

def run_experiment1():
    data = pd.read_csv('data/dataset.csv')
    model = create_bayesian_model(data)
    with model:
        trace = pm.sample(1000, return_inferencedata=True)
    return trace

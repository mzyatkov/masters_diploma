import pymc as pm
import numpy as np
    
def create_bayesian_model(data):
    with pm.Model() as model:
        alpha = pm.Exponential('alpha', 1.0)
        beta = pm.Exponential('beta', 1.0)
        lambda_ = pm.Gamma('lambda_', alpha, beta)
        
        time_to_failure = pm.Exponential('time_to_failure', lambda_, observed=data['time_to_failure'])
        
    return model

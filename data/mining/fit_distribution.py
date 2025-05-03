import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


# 1. Load and validate data
def load_and_validate_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic validation: no missing manufacturers or model identifiers
    required_cols = ['mfg', 'model', 'days_alive', 'cost_estimate']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Drop rows with critical nulls
    df = df.dropna(subset=required_cols)
    return df


# 2. Group by manufacturer and model
def group_data(df: pd.DataFrame):
    return {
        (mfg, model): group.reset_index(drop=True)
        for (mfg, model), group in df.groupby(['mfg', 'model'])
    }


# 3. Fit Bayesian model for each group
#    Example: modeling days_alive ~ Exponential(lambda) and cost_estimate ~ Normal(mu, sigma)
def fit_group_model(group_df: pd.DataFrame):
    with pm.Model() as model:
        # Priors
        lambda_prior = pm.HalfNormal('lambda', sigma=1.0)
        mu_cost = pm.Normal('mu_cost', mu=group_df['cost_estimate'].mean(), sigma=10)
        sigma_cost = pm.HalfNormal('sigma_cost', sigma=10)

        # Likelihoods
        days_obs = pm.Exponential('days_alive', lam=lambda_prior, observed=group_df['days_alive'])
        cost_obs = pm.Normal('cost_estimate', mu=mu_cost, sigma=sigma_cost, observed=group_df['cost_estimate'])

        # Inference
        trace = pm.sample(draws=1000, tune=1000, cores=2, target_accept=0.9)
    return model, trace


# 4. Posterior predictive sampling
#    Use the fitted trace to sample new failure and cost values

def posterior_predictive(model, trace, samples: int = 1000):
    with model:
        ppc = pm.sample_posterior_predictive(trace, var_names=['days_alive', 'cost_estimate'])
    return ppc


# Main orchestration
def main(data_path: str):
    df = load_and_validate_data(data_path)
    groups = group_data(df)

    results = {}
    for key, grp in groups.items():
        print(f"Fitting model for {key}")
        model, trace = fit_group_model(grp)
        ppc = posterior_predictive(model, trace, samples=500)
        results[key] = {'model': model, 'trace': trace, 'ppc': ppc}

        plt.clf()
        pm.plot_trace(trace)
        plt.savefig(f"./figures/{key}.png")

    return results


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else './output/train.csv'
    results = main(path)
    # results contains posterior traces and predictive samples per group
    print('Done.')

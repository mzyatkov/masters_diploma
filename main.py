from experiments.experiment1 import run_experiment1
from experiments.experiment2 import run_experiment2
import arviz as az
import pandas as pd

def main():
    print("Running Experiment 1...")
    inference_data = run_experiment1()
    az.to_netcdf(inference_data, 'results/bayesian_trace.nc')
    print("Experiment 1 completed and results saved.")
    
    print("Running Experiment 2...")
    result = run_experiment2()
    print("Experiment 2 completed.")
    result.history = {'n_gen': [i for i in range(len(result.history))], 'opt': [h['opt'] for h in result.history]}
    pd.to_pickle(result, 'results/optimization_result.pkl')
    print("Experiment 2 completed and results saved.")

if __name__ == "__main__":
    main()

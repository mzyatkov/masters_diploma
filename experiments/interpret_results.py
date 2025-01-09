import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

def interpret_bayesian_results(trace):
    print("Summary of Bayesian Model:")
    summary = az.summary(trace)
    print(summary)
    
    az.plot_trace(trace)
    plt.show()

def interpret_optimization_results(result):
    print("Optimization Result:")
    print("Best solution found:", result.X)
    print("Function value at best solution:", result.F)
    
    # Example of plotting convergence
    plt.plot(result.history['n_gen'], result.history['opt'])
    plt.xlabel('Generation')
    plt.ylabel('Objective Value')
    plt.title('Convergence Plot')
    plt.show()

def main():
    # Load results from experiments
    # For demonstration, let's assume we have some saved results
    # In practice, you would load these from files or pass them directly
    bayesian_trace = az.from_netcdf('results/bayesian_trace.nc')
    optimization_result = pd.read_pickle('results/optimization_result.pkl')
    
    interpret_bayesian_results(bayesian_trace)
    interpret_optimization_results(optimization_result)

if __name__ == "__main__":
    main()

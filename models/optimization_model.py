from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems import get_problem

def optimize_schedule():
    problem = get_problem("rastrigin")
    algorithm = GA(pop_size=100)
    res = minimize(problem, algorithm, termination=('n_gen', 50))
    return res

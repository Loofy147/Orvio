import numpy as np
from pipeline_agent import NumericOptimizer

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)

def main():
    d = 2
    bounds = np.array([[-2.0, 2.0]] * d)
    agent = NumericOptimizer(rosenbrock, bounds, max_rounds=10)
    result = agent.run()
    print(f"Best score: {result['best_score']}")
    print(f"Eval count: {result['eval_count']}")

if __name__ == "__main__":
    main()

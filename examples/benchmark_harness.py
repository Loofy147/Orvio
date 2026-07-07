import numpy as np
import time
from numeric_optimizer import SolverOrchestrator

def sphere(x): return np.sum(x**2)
def rosenbrock(x): return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)
def ackley(x):
    d = len(x)
    return -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / d) + 20 + np.e

def run_benchmark(name, fn, d, bounds, budget=2000, seeds=[0, 1, 2]):
    print(f"\n=== Benchmarking {name} (d={d}) ===")
    scores = []
    evals_list = []
    t0 = time.time()

    for seed in seeds:
        orch = SolverOrchestrator(fn, bounds, budget=budget, seed=seed)
        res = orch.run()
        scores.append(res['best_score'])
        evals_list.append(res['evals'])

    duration = time.time() - t0
    print(f"Results for {name}:")
    print(f"  Mean Score: {np.mean(scores):.6e} (+/- {np.std(scores):.2e})")
    print(f"  Best Score: {np.min(scores):.6e}")
    print(f"  Mean Evals: {np.mean(evals_list):.1f}")
    print(f"  Total Time: {duration:.2f}s")

def main():
    # 2D Problems
    bounds_2d = np.array([[-5.0, 5.0]] * 2)
    run_benchmark("Sphere 2D", sphere, 2, bounds_2d)
    run_benchmark("Rosenbrock 2D", rosenbrock, 2, bounds_2d)
    run_benchmark("Ackley 2D", ackley, 2, bounds_2d)

    # 5D Problems
    bounds_5d = np.array([[-5.0, 5.0]] * 5)
    run_benchmark("Sphere 5D", sphere, 5, bounds_5d, budget=5000)

if __name__ == "__main__":
    main()

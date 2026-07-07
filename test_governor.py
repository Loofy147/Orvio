import numpy as np
from pipeline_agent import Governor

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)

def noisy_sphere(x):
    return np.sum(x**2) + np.random.normal(0, 0.1)

def main():
    bounds = np.array([[-2.0, 2.0]] * 2)

    print("--- Testing Governor on Rosenbrock (Deterministic) ---")
    gov1 = Governor(rosenbrock, bounds, budget=2000)
    res1 = gov1.run()
    print(f"Best score: {res1['best_score']}")
    print(f"Modes run: {res1['modes']}")
    print(f"Evals used: {res1['evals']}")

    print("\n--- Testing Governor on Noisy Sphere ---")
    gov2 = Governor(noisy_sphere, bounds, budget=2000)
    res2 = gov2.run()
    print(f"Best score: {res2['best_score']}")
    print(f"Modes run: {res2['modes']}")
    print(f"Evals used: {res2['evals']}")

if __name__ == "__main__":
    main()

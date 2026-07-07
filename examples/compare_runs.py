import numpy as np
import json
from numeric_optimizer import SolverOrchestrator

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)

def compare_demo():
    bounds = np.array([[-2.0, 2.0]] * 2)

    print("Running Experiment 1: Budget 1000")
    orc1 = SolverOrchestrator(rosenbrock, bounds, budget=1000, seed=42)
    res1 = orc1.run()
    orc1.save_report("run1.json")

    print("Running Experiment 2: Budget 3000")
    orc2 = SolverOrchestrator(rosenbrock, bounds, budget=3000, seed=42)
    res2 = orc2.run()
    orc2.save_report("run2.json")

    with open("run1.json") as f: r1 = json.load(f)
    with open("run2.json") as f: r2 = json.load(f)

    print("\n=== SIDE-BY-SIDE COMPARISON ===")
    print(f"Metric        | Run 1 (B=1000) | Run 2 (B=3000)")
    print(f"--------------|----------------|---------------")
    print(f"Best Score    | {r1['best_score']:.6e} | {r2['best_score']:.6e}")
    print(f"Total Evals   | {sum(p['eval_count'] for p in r1['portfolio']):14} | {sum(p['eval_count'] for p in r2['portfolio']):14}")
    print(f"Modes Used    | {len(r1['portfolio']):14} | {len(r2['portfolio']):14}")

if __name__ == "__main__":
    compare_demo()

import numpy as np
from pipeline_agent import Governor

def discrete_sphere(x):
    # Rounded values to simulate discrete domain
    return np.sum(np.round(x)**2)

def main():
    # Only integers in [0, 10]
    bounds = np.array([[0.0, 10.0]] * 2)

    # We force the input to be integer in the objective to simulate discrete
    # But for TaskTyper to detect it, the probes (which are continuous) must
    # already look discrete, or we need a better heuristic.
    # In a real scenario, the user would provide an objective that only makes sense on integers.

    print("--- Testing Governor on Discrete Sphere ---")
    gov = Governor(discrete_sphere, bounds, budget=1000)
    res = gov.run()
    print(f"Best score: {res['best_score']}")
    print(f"Best x: {res['best_x']}")

if __name__ == "__main__":
    main()

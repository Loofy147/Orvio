import numpy as np
from pipeline_agent import NumericOptimizer

def main():
    d = 2
    bounds = np.array([[-5.0, 5.0]] * d)
    # Sphere function: Exploring and Optimizing should be very useful initially
    agent = NumericOptimizer(lambda x: np.sum((x-1)**2), bounds, max_rounds=15)
    result = agent.run()

    print("Final budget parameters:")
    print(f"explore_n: {agent.adaptive_explore_n}")
    print(f"sleuth_samples: {agent.adaptive_sleuth_samples}")
    print(f"opt_rounds: {agent.opt_rounds}")
    print(f"ft_rounds: {agent.ft_rounds}")
    print(f"honing_samples: {agent.honing_samples}")

    for p, s in result['phase_stats'].items():
        if s['evals'] > 0:
            print(f"{p}: util = {s['improvement']/s['evals']:.4f}")

if __name__ == "__main__":
    main()

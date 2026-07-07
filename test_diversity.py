import numpy as np
from pipeline_agent import PipelineAgent

def multi_modal(x):
    # Multiple local minima
    return np.sum(np.sin(5*x) + x**2)

def main():
    d = 2
    bounds = np.array([[-2.0, 2.0]] * d)
    # Low max_rounds to force multiple restarts if we keep it small or mock stalling
    agent = PipelineAgent(multi_modal, bounds, max_rounds=20, tol=1e-1)
    result = agent.run()

    print(f"Basins archived: {len(agent.basin_archive)}")
    for i, b in enumerate(agent.basin_archive):
        print(f"Basin {i}: {b}")

if __name__ == "__main__":
    main()

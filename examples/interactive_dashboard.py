import numpy as np
import time
import threading
import sys
from numeric_optimizer import NumericOptimizer, OptimizationGoal, QuadraticSurrogate

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)

def dashboard_demo():
    print("=== ORVIO INTERACTIVE STEERING DEMO ===")

    goal = OptimizationGoal(
        objective=rosenbrock,
        bounds=np.array([[-2.0, 2.0]] * 2)
    )

    agent = NumericOptimizer(goal, budget=5000, max_rounds=50)

    # Run the agent in a background thread
    def run_agent():
        for _ in agent.run_iterator():
            pass
        print("\n[AGENT] Optimization complete.")

    thread = threading.Thread(target=run_agent)
    thread.start()

    print("Agent is running. Options: [p]ause, [r]esume, [d]isable phase, [b]udget boost, [q]uit")

    try:
        while thread.is_alive():
            # In a real UI, this would be a reactive state.
            # Here we just show status and take a single command at a time.
            state = agent.get_state()
            print(f"\r[STATUS] Phase: {state['current_phase'] or '...':12} | Best: {state['best_score']:.6f} | Evals: {state['evals']:4} | Radius: {state['radius']:.4f}", end="")
            sys.stdout.flush()

            # Simple simulation of user interaction after 1 second
            time.sleep(1.5)

            print("\n[USER] Intervening: Pausing agent...")
            agent.manual_pause = True
            time.sleep(1)

            print("[USER] Disabling 'Fine-tuning' phase to save budget.")
            agent.disabled_phases.add("Fine-tuning")

            print("[USER] Doubling budget mid-run.")
            agent.budget *= 2

            print("[USER] Resuming agent...")
            agent.manual_pause = False

            # Wait for more progress or completion
            thread.join(timeout=10)
            break

    except KeyboardInterrupt:
        pass

    print("\n=== DEMO COMPLETE ===")
    final_res = agent._finalize()
    print(f"Final Score: {final_res['best_score']:.8e}")
    print(f"Total Evals: {final_res['eval_count']} / {agent.budget}")
    print(f"Disabled Phases: {agent.disabled_phases}")

if __name__ == "__main__":
    dashboard_demo()

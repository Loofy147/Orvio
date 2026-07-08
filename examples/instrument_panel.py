import numpy as np
import time
import sys
from numeric_optimizer import SolverOrchestrator, OptimizationGoal

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)

def instrument_panel_demo():
    print("=== ORVIO INSTRUMENT PANEL DEMO ===")

    # Define the Goal
    goal = OptimizationGoal(
        objective=rosenbrock,
        bounds=np.array([[-2.0, 2.0]] * 2),
        target_score=1e-4,
        success_criteria=lambda score, x: score < 5e-5
    )

    print(f"Goal: Minimize Rosenbrock 2D")
    print(f"Target Score: {goal.target_score} | Success Criteria: score < 5e-5")

    # Orchestrate with Parallelism option
    orchestrator = SolverOrchestrator(
        goal, budget=2000, parallel=False # Sequential for better live visualization
    )

    print("\n[INIT] Characterizing problem...")

    it = orchestrator.run_iterator()

    for msg in it:
        mtype = msg["type"]

        if mtype == "characterization":
            d = msg["data"]
            print(f"[EVIDENCE] d={d['d']}, is_det={d['is_deterministic']}, ruggedness={d['ruggedness']:.2f}")

        elif mtype == "mode_start":
            print(f"\n[DECISION] Switching to Mode: {msg['mode']}")

        elif mtype == "phase":
            state = msg["state"]
            round_num = msg["round"]
            phase_name = msg["phase"]
            sys.stdout.write(f"\rRound {round_num:2} | Phase: {phase_name:12} | Best Score: {state['best_score']:.6f} | Evals: {state['evals']:4}")
            sys.stdout.flush()

            if phase_name == "Iterating" and state["log"]:
                last_log = state["log"][-1]
                if "basin_hop" in last_log.summary.get("action", ""):
                    print(f"\n[CONTROL] {last_log.summary.get('reason', '')}")

            time.sleep(0.01)

        elif mtype == "mode_end":
            print(f"\n[RESULT] Mode {msg['mode']} finished. Portfolio best: {msg['best_score']:.6e}")

    final = orchestrator.run()
    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Final Score: {final['best_score']:.8e}")
    print(f"Total Evals: {final['evals']}")

    orchestrator.save_report("demo_report.json")
    print("\n[EXPORT] Detailed report saved to demo_report.json")

if __name__ == "__main__":
    instrument_panel_demo()

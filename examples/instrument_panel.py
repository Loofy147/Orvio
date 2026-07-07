import numpy as np
import time
import sys
from numeric_optimizer import SolverOrchestrator

def rosenbrock(x):
    # Success criterion: find a score < 0.01
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)

def instrument_panel_demo():
    print("=== ORVIO INSTRUMENT PANEL DEMO ===")
    print("Defining Goal: Optimize Rosenbrock 2D")
    print("Bounds: [[-2, 2], [-2, 2]]")
    print("Target Score: 1e-4 (Early Exit Success Criterion)")

    bounds = np.array([[-2.0, 2.0]] * 2)

    # Choose the mode: We'll let the orchestrator decide, but we'll watch it.
    orchestrator = SolverOrchestrator(
        rosenbrock, bounds, budget=2000, target_score=1e-4
    )

    print("\n[INIT] Characterizing problem...")

    it = orchestrator.run_iterator()

    current_mode = ""
    for msg in it:
        mtype = msg["type"]

        if mtype == "characterization":
            d = msg["data"]
            print(f"[EVIDENCE] d={d['d']}, is_det={d['is_deterministic']}, ruggedness={d['ruggedness']:.2f}")

        elif mtype == "mode_start":
            current_mode = msg["mode"]
            print(f"\n[DECISION] Switching to Mode: {current_mode}")

        elif mtype == "phase":
            state = msg["state"]
            round_num = msg["round"]
            phase_name = msg["phase"]

            # Show the process: each phase, each round
            sys.stdout.write(f"\rRound {round_num:2} | Phase: {phase_name:12} | Best Score: {state['best_score']:.6f} | Evals: {state['evals']:4}")
            sys.stdout.flush()

            # If Iterating, check the reason
            if phase_name == "Iterating":
                reason = state["log"][-1].summary.get("reason", "")
                if "basin_hop" in state["log"][-1].summary.get("action", ""):
                    print(f"\n[CONTROL] {reason}")

            # Slow down for demo effect
            time.sleep(0.02)

        elif mtype == "mode_end":
            print(f"\n[RESULT] Mode {msg['mode']} finished. Portfolio best: {msg['best_score']:.6e}")

    final = orchestrator.run()
    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Final Score: {final['best_score']:.8e}")
    print(f"Total Evals: {final['evals']}")
    print(f"Portfolio Used: {final['modes']}")

    # Export evidence
    orchestrator.save_report("demo_report.json")
    print("\n[EXPORT] Detailed report saved to demo_report.json")

if __name__ == "__main__":
    instrument_panel_demo()

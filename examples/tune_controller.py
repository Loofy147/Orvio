import numpy as np
from numeric_optimizer import SolverOrchestrator

def simulate_system(pid_params, steps=100, target=1.0):
    """
    Simulates a simple mass-spring-damper system with a PID controller.
    pid_params: [Kp, Ki, Kd]
    Returns Integral Absolute Error (IAE).
    """
    Kp, Ki, Kd = pid_params
    pos = 0.0
    vel = 0.0
    integral = 0.0
    prev_error = 0.0
    iae = 0.0

    dt = 0.1
    mass = 1.0
    b = 0.5 # damping
    k = 2.0 # spring

    for _ in range(steps):
        error = target - pos
        integral += error * dt
        derivative = (error - prev_error) / dt

        force = Kp * error + Ki * integral + Kd * derivative

        # Physics: mass*acc + b*vel + k*pos = force
        acc = (force - b * vel - k * pos) / mass
        vel += acc * dt
        pos += vel * dt

        iae += abs(error) * dt
        prev_error = error

        # Stability check
        if abs(pos) > 100: return 1e6

    return iae

def main():
    # Bounds for [Kp, Ki, Kd]
    bounds = np.array([[0.0, 20.0], [0.0, 10.0], [0.0, 10.0]])

    print("--- Tuning PID Controller for Mass-Spring-Damper ---")
    orchestrator = SolverOrchestrator(simulate_system, bounds, budget=1000)
    result = orchestrator.run()

    print("\nOptimization Report:")
    print("\n".join(result['optimization_report']))

    print(f"\nBest PID parameters: {result['best_x']}")
    print(f"Best IAE: {result['best_score']:.4f}")

if __name__ == "__main__":
    main()

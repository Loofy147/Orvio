import numpy as np
from epistem import EpistemEngine, generate_report

def main():
    options = [
        "Experienced backend engineer, 10 years experience, specializes in high-scale distributed systems and Python performance.",
        "Talented junior dev, recent bootcamp grad, extremely high potential, very eager to learn, strong culture fit, low salary requirement.",
        "Solid mid-level fullstack developer, reliable, good communicator, experience with React and Node, moderate salary.",
        "Academic researcher, PhD in AI, very deep theoretical knowledge, limited industry experience, high salary requirement."
    ]
    option_names = ["Senior Backend", "Junior Rising Star", "Reliable Fullstack", "AI Researcher"]
    party_names = ["Engineering Manager", "Product Lead", "CFO", "CTO"]

    # 4 parties, 4 dimensions
    party_weights = np.array([
        [0.8, 0.2, 0.0, 0.5], # EM
        [0.3, 0.7, 0.2, 0.1], # PL
        [0.1, 0.1, 0.9, 0.0], # CFO
        [0.5, 0.0, 0.1, 0.9]  # CTO
    ])

    # Normalize party weights
    party_weights = party_weights / party_weights.sum(axis=1)[:, None]

    print("--- Running Epistem Hiring Committee Consensus ---")
    engine = EpistemEngine(option_names, options, party_weights, party_names)
    consensus, stress_report = engine.run()

    generate_report(consensus, stress_report)

if __name__ == "__main__":
    main()

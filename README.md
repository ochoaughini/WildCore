# WildCore: Advanced AI Security & Anomaly Detection Framework

![CI/CD](https://github.com/ochoaughini/WildCore/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**WildCore is an open-source framework for simulating, detecting, and analyzing vulnerabilities in embedding-based Artificial Intelligence systems.**

This repository provides tools for security researchers, ML engineers, and developers to prototype and validate defenses against threats like prompt injection, vector poisoning, and anomalous AI agent behavior.

---

## ✨ Key Features

*   **Simulation Agent (`GutoVectorWildcard`):** A simulated AI agent capable of deviating from containment protocols, serving as a "red team" to test defenses.
*   **Self-Regulated Detector (`AutoRegulatedPromptDetector`):** A multi-layered defense system that combines multiple detection techniques and adjusts its own parameters in real-time.
*   **Ensemble Detection:** Utilizes a voting system among different methods (cosine similarity, anomaly scoring, unsupervised learning) to increase accuracy and reduce false positives.
*   **Realistic Simulation:** Includes scripts to generate embedding datasets that simulate normal, suspicious, and malicious behavior.
*   **Ready for Extension:** Modular structure designed for the easy addition of new detectors and agents.

---

## 🚀 Quick Start & Installation

To run the main WildCore demonstration, follow these steps.

**1. Clone the repository:**
```bash
git clone https://github.com/ochoaughini/WildCore.git
cd WildCore
```

**2. Create a virtual environment and install dependencies:**
```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

**3. Run the demonstration:**
```bash
python main.py
```

You will see the simulation running in the terminal, showing detections at each iteration and a final system performance summary.

## 🛠️ Project Structure

The project is organized modularly to facilitate maintenance and contributions:
```
WildCore/
├── wildcore/          # Main WildCore module
│   ├── agent.py       # Defines GutoVectorWildcard
│   └── detector.py    # Defines AutoRegulatedPromptDetector
├── tests/             # Unit tests
├── main.py            # Entry point for the demonstration
└── ...
```

## 🤝 How to Contribute

Contributions are very welcome! If you have ideas for new features, detectors, or improvements, please read our Contribution Guide.

## 📜 License

This project is licensed under the MIT License.

# Self‑Localizing Sensor Network (SLSN)

<p align="center">
  <video width="600" controls autoplay muted loop>
    <source src="results/localization.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>

Recover the relative geometry of a random sensor deployment using only inter‑node distance measurements – no GPS, no centralised localisation infrastructure. The pipeline combines:

- **Anchor selection** – picks three well‑spaced nodes with direct links.
- **Canonical placement** – fixes anchors in a reference frame.
- **Iterative trilateration** – localises nodes with ≥3 known neighbours.
- **Stress minimisation** – refines all free nodes via gradient descent on the mean squared edge‑length error.

The code is fully self‑contained, with visualisation tools to inspect convergence, error distributions, and network connectivity.

---

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/yourusername/self-localizing-sensor-network.git
cd self-localizing-sensor-network
pip install -e .
```

For running tests, install the optional test dependencies:
```bash
pip install -e .[test]
```

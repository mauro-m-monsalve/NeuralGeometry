<table>
<tr>
<td><img src="logo.png" width="200"/></td>
<td>
  <h2>NeuralGeometry</h2>
  <p>
    Code, data, and analysis for the paper<br/>
    <strong>“The geometry of the neural state space of decisions”</strong><br/>
    <em>(Monsalve-Mercado et al., 2025)</em><br/>
    <a href="https://doi.org/10.1101/2025.01.24.634806">https://doi.org/10.1101/2025.01.24.634806</a>
  </p>
</td>
</tr>
</table>

---

### Overview

This repository provides tools to analyze high-dimensional neural population activity during decision-making tasks.  
It focuses on uncovering the geometric structure of neural dynamics, including manifold learning, single-trial projections, and dynamical signatures across conditions.

The analysis covers neural manifold construction, arc-length parameterization, reaction-time alignment, and decomposition of neural trajectories into meaningful geometric components.

[https://doi.org/10.1101/2025.01.24.634806](https://doi.org/10.1101/2025.01.24.634806)

---

### Processed Data

All datasets are preprocessed and available via Zenodo:  
[doi:10.5281/zenodo.15093134](https://doi.org/10.5281/zenodo.15093134)  
These files include firing rates, LFADS-inferred trajectories, trial metadata, and behavioral annotations.

---

### Interactive Notebooks

Launch the full repository in an interactive Binder session:  
[mybinder.org/v2/gh/mauro-m-monsalve/NeuralGeometry/main](https://mybinder.org/v2/gh/mauro-m-monsalve/NeuralGeometry/main)

Notebooks are located in the `notebooks/` folder. Open a notebook and click run -> run all cells to generate figures.

---

### What can you do with this repository?

- Load and align population activity across sessions, conditions, and behavioral measures.
- Construct low-dimensional decision manifolds using PCA and arc-length parameterization.
- Smooth and interpolate neural trajectories across reaction time or evidence strength.
- Decompose single-trial activity into components aligned with manifold geometry:
  - Resolution direction (arc-length tangent)
  - Uncertainty direction (transverse to arc-length)
  - Off-manifold deviations
- Compute curvature and tortuosity of neural state space trajectories.
- Visualize population geometry using high-resolution, interactive 2D and 3D tools.

---

### Citation

If you use this code or dataset, please cite:

> Monsalve-Mercado et al. (2025).  
> *The geometry of the neural state space of decisions*.  
> [https://doi.org/10.1101/2025.01.24.634806](https://doi.org/10.1101/2025.01.24.634806)

---

### Quick Links

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15093134.svg)](https://doi.org/10.5281/zenodo.15093134)  
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mauro-m-monsalve/NeuralGeometry/main)

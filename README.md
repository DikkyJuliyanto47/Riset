# MADRL for Flood Evacuation Routing  
### Case Study: Surabaya City, Indonesia

This repository contains an experimental **Multi-Agent Deep Reinforcement Learning (MADRL)** framework for **flood evacuation route selection**, developed as part of an ongoing disaster mitigation research project in Indonesia.

The system models **multiple evacuees (agents)** that simultaneously select evacuation points under flood constraints, distance efficiency, elevation safety, and congestion effects.

---

## 🔍 Research Context

- **Hazard type**: Flood
- **Study area**: Surabaya City, East Java, Indonesia
- **Approach**: One-shot Multi-Agent Reinforcement Learning
- **Training scheme**: Centralized Training – Decentralized Execution (CTDE)
- **Focus**: Decision-level evacuation point selection (not path planning)

This work is an extension of earlier evacuation routing studies using **KNN and single-agent RL**, now expanded into a **multi-agent setting**.

---

## 🧠 Core Idea

Each agent represents a potential evacuee and must choose **one evacuation point** based on:

- Distance to evacuation point (OSRM-based)
- Flood exposure (polygon-based flood map)
- Elevation difference (DEM raster)
- Congestion (number of agents choosing the same evacuation point)

The agents are trained jointly using a centralized critic to encourage  
**safe, efficient, and distributed evacuation decisions**.

---

## 🏗️ Project Structure

```text
.
├── conda/                 # Conda environment definition
│   └── environment.yml
├── dataset/               # Dataset files (not tracked by git)
├── madrI_config.py        # Global configuration & hyperparameters
├── madrI_env.py           # One-shot multi-agent environment
├── madrI_models.py        # Actor & Centralized Critic networks
├── madrI_trainer.py       # MADRL training loop
├── madrI_evaluator.py     # Model evaluation & testing
├── madrI_utils.py         # OSRM, flood check, elevation utilities
├── training.ipynb         # Jupyter notebook for experiments
├── .gitignore
└── README.md
```

## ⚙️ Installation (Conda Environment) — REQUIRED FIRST STEP

⚠️ **Before running any code or notebook, you MUST install the Conda environment provided in the `conda/` folder.**

This project relies on geospatial and deep learning libraries that are best managed via Conda.

### 1️⃣ Create a New Conda Environment

From the project root directory, run:

```bash
conda env create -f conda/environment.yml
````

This will create a new Conda environment named **`madrl`**.

---

### 2️⃣ Activate the Environment

```bash
conda activate madrl
```

---

### 3️⃣ Register Jupyter Kernel (Recommended)

To use the environment inside Jupyter Notebook:

```bash
python -m ipykernel install --user --name=madrl --display-name="MADRL Env"
```

Then in Jupyter Notebook:

> **Kernel → Change Kernel → MADRL Env**

---

### 📦 Conda Environment Specification

The environment is defined in:

```text
conda/environment.yml
```

Contents:

```yaml
name: madrl
channels:
  - pytorch
  - conda-forge
  - defaults

dependencies:
  - python=3.10
  - pytorch
  - torchvision
  - torchaudio
  - cpuonly
  - numpy
  - pandas
  - geopandas
  - rasterio
  - shapely
  - requests
  - matplotlib
  - scikit-learn
  - pip
  - pip:
      - contextily
      - tqdm
      - ipykernel
```

> 💡 **GPU users**: replace `cpuonly` with `cudatoolkit=11.8` (or compatible CUDA version).

---
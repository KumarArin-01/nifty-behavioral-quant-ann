# nifty-behavioral-quant-ann
A multi-modal deep learning model using Keras/TensorFlow to forecast Nifty 50 directional transitions using alternative data (Lunar synodic cycles &amp; meteorological vectors).

# Multi-Modal Deep Learning Architecture for Market Directional Forecasting Using Alternative Exogenous Data

## 🌌 Project Abstract
This repository contains an algorithmic feedforward Artificial Neural Network (ANN) built using Keras and TensorFlow to forecast the directional volatility of the India Nifty 50 Index over a 1-day horizon. 

Instead of relying on traditional lagging technical indicators (like RSI or MACD) which are prone to market whipsaws, this framework explores an **orthogonal multi-modal pipeline**. It models market psychology by fusing localized meteorological climate dynamics (Meteostat API - Santacruz Observatory, Mumbai) and astrophysical continuous lunar synodic wave vectors (29.53059-day cycle) alongside spot gold market asset flows (`yfinance`).

---

## 🛠️ Feature Engineering & Data Modalities

The network processes five structural leading indicators to predict if the market will close **UP (1)** or **DOWN (0)** on the next trading session:

1. **Nifty_Return_1D:** Daily logarithmic percentage returns of the Nifty 50 (`^NSEI`) to capture baseline momentum.
2. **Gold_Return:** Logarithmic daily returns of COMEX Gold Futures (`GC=F`) to monitor institutional risk-off hedging shifts.
3. **Gold_Nifty_Ratio:** A cross-asset macro metric monitoring systemic capital rotation between equities and safe-haven commodities.
4. **Lunar_Phase_Wave:** A continuous, zero-lag Cosine wave transformation tracking the exact human biological circadian rhythm shifts mapped to the 29.53059-day synodic month.
5. **tavg:** Daily baseline ambient temperature recorded directly at the Santacruz Observatory in Mumbai, isolating systemic weather disruptions affecting core local financial institutional desks.

---

## 🧠 Model Architecture & Regularization

Because alternative behavioral features have an incredibly low signal-to-noise ratio, standard neural networks easily fall into overfitting or complacency traps. This architecture mitigates those traps through the following design choices:

* **Layer Layout:** A sequential feedforward structure consisting of an input layer, a 32-neuron dense layer (ReLU), a 15% Dropout layer to kill lazy co-dependencies, a 16-neuron dense layer, and a final 1-neuron Sigmoid output layer for binary probability matching.
* **Cost-Sensitive Learning (Class Weights):** Equity indices naturally possess a long-term upward drift, which causes native models to over-predict "Up" days. This pipeline computes the exact mathematical inverse frequency of the target labels during training. It heavily penalizes missed Down-market signals, forcing the Adam optimizer to establish bidirectional tracking boundaries.

---

## 🚀 Installation & Local Replication

### 1. Setup Active Workspace
Clone this repository to your local system and install the pinned, conflict-free package dependencies within an isolated virtual environment:

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/nifty-behavioral-quant-ann.git](https://github.com/YOUR_USERNAME/nifty-behavioral-quant-ann.git)
cd nifty-behavioral-quant-ann

# Create and activate environment
python -m venv quant_env
source quant_env/bin/activate  # On Windows use: quant_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

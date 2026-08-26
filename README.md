# Specialized Project

A comprehensive deep learning and algorithmic trading platform for financial analysis and investment strategies.

## Project Overview

This project implements a sophisticated trading system that combines traditional machine learning approaches with factor-based investing models. The platform provides tools for data ingestion, model training, evaluation, and portfolio management with a focus on systematic investment strategies.

## Workspace Structure

```
Specialized Project/
├── .venv/                          # Python virtual environment
├── .vscode/                        # VS Code configuration
├── artifacts/                      # Generated artifacts and outputs
│   ├── checkpoints/                # Model checkpoints
│   ├── oos/                        # Out of sample (OOS) prediction
├── db/                             # S&P500 daily OHLCV data
├── docs/                           # LaTeX documentation
├── factors/                        # Fama-French data
├── logs/                           # Log files
├── notebooks/                      # Jupyter notebooks for experimentation
├── refs/                           # Reference materials and documentation
├── src/                            # Main source code
│   ├── broker/                     # Broker API integrations and data ingestion (Phase 2)
│   ├── data/                       # Data ingestor and processing
│   │   ├── alpaca_ingestor.py      # Alpaca data ingestion
│   │   ├── residualizer.py         # Residual calculation utilities
│   │   ├── splitters.py            # Data splitting utilities
│   │   ├── yf_ingestor.py          # Yahoo Finance data ingestion
│   │   └── client.py               # Broker client interface
│   ├── evaluation/                 # Model evaluation and backtesting
│   ├── models/                     # Machine learning models
│   ├── portfolio/                  # Portfolio management and optimization (Phase 2)
│   ├── training/                   # Model training pipelines
│   ├── config.py                   # Project configuration
│   ├── modal_run.py                # Modal run script
│   ├── modal_upload.py             # Modal upload script
│   └── train.py                    # Main training script
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables file
└── .gitignore                      # Git ignore patterns
```

## Key Features

- **Multiple Data Sources**: Integrated data ingestion from Alpaca and Yahoo Finance
- **Factor-Based Models**: Built-in factor calculation and embedding capabilities
- **Model Training**: Comprehensive ML training pipelines with custom optimizers
- **Evaluation Framework**: Detailed model evaluation and backtesting
- **Portfolio Optimization**: Advanced portfolio construction and risk management
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Documentation**: Comprehensive docs and examples

## Setup Instructions

### Prerequisites

- Python 3.11
- pip (Python package installer) or [uv](https://github.com/astral-sh/uv)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/doxncwfn/QuantAlphaMLOps.git
cd Specialized\ Project
```

1. **Create virtual environment**

```bash
uv venv --python 3.11.7
```

1. **Install dependencies**

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

1. **Configure environment**

Create a `.env` file in the project root using the sample provided in `.env.example`:

```env
<!-- TOBEDONE -->
```

## Usage

### Data Ingestion Pipeline

```bash
# Download S&P500 Daily OHLCV Data
python src/data/yf_ingestor.py

# Download & cache Fama-French 3 Factors
python src/data/residualizer.py
```

### Training a Model

```bash
# TO be completed
```

### Running Evaluations

```bash
# To be completed
```

### Portfolio Analysis

```python
# To be completed
```

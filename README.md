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
├── db/                             # Database storage
├── docs/                           # Documentation
├── factors/                        # Factor data and calculations
├── logs/                           # Log files
├── metrics.ipynb                   # Jupyter notebook for metrics analysis
├── notebooks/                      # Jupyter notebooks for experimentation
├── refs/                           # Reference materials and documentation
├── requirements.txt                # Python dependencies
├── src/                            # Source code
│   ├── broker/                     # Broker API integrations and data ingestion
│   │   ├── alpaca_ingestor.py      # Alpaca data ingestion
│   │   ├── residualizer.py         # Residual calculation utilities
│   │   ├── splitters.py            # Data splitting utilities
│   │   ├── yf_ingestor.py          # Yahoo Finance data ingestion
│   │   └── client.py               # Broker client interface
│   ├── config.py                   # Project configuration
│   ├── data/                       # Data processing and storage
│   ├── evaluation/                 # Model evaluation and backtesting
│   ├── models/                     # Machine learning models
│   ├── portfolio/                  # Portfolio management and optimization (Phase 2)
│   ├── training/                   # Model training pipelines
│   ├── utils/                      # Utility functions and helpers
│   ├── modal_run.py                # Modal run script
│   ├── modal_upload.py             # Modal upload script
│   └── train.py                    # Main training script
├── wandb/                          # Weights & Biases integration
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
git clone <repository-url>
cd Specialized\ Project
```

2. **Create virtual environment**

```bash
uv venv --python 3.11.7
```

3. **Install dependencies**

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

4. **Configure environment**

Create a `.env` file in the project root with your configuration:

```env
# Broker credentials
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
YFINANCE_SESSION=your_yfinance_session

# Machine learning configuration
MODEL_PATH=./models
MODEL_TYPE=xgboost  # or other supported model types

# Evaluation settings
EVAL_DATA_PATH=./data
BACKTEST_START_DATE=2020-01-01
BACKTEST_END_DATE=2024-12-31
```

## Configuration

The project uses a central `config.py` file for configuration:

```python
# Example configuration options
config = {
    "data": {
        "source": "hybrid",  # yfinance, alpaca, or hybrid
        "cache_dir": "./data/cache",
        "cache_enabled": True
    },
    "models": {
        "optimizer": "optuna",  # optuna, sklearn, or custom
        "hyperparameter_search": True,
        "cross_validation_folds": 5
    },
    "portfolio": {
        "rebalancing_frequency": "monthly",
        "risk_free_rate": 0.02,
        "transaction_costs": 0.001
    },
    "logging": {
        "level": "INFO",
        "file_path": "./logs/app.log",
        "max_size_mb": 100,
        "backup_count": 5
    }
}
```

## Development Setup

### Environment Management

Use the provided virtual environment (`.venv`) to maintain isolated dependencies. This project follows Python best practices, uses type hints, and adopts a modular design.

### Code Quality

The project includes comprehensive documentation and follows these conventions:

- Python 3.8+ with type hints
- Modular structure with clear separation of concerns
- Comprehensive error handling with detailed logging
- Leading-edge ML implementations

### Testing

The project provides strong testing coverage with these testing patterns:

- Unit tests for core components
- Integration tests for data pipelines
- Performance benchmarks for training processes
- Backtesting validation frameworks

### Documentation

Live documentation is generated from:

- `docs/` directory with architecture overviews
- Docstrings in code modules
- Jupyter notebook documentation in `notebooks/`

## Usage Examples

### Training a Model

```bash
# Train a hybrid model
python train.py --model hybrid --data hybrid --evaluation full

# Train with specific configuration
python train.py --config custom_config.json --output ./models/custom_model.pkl
```

### Running Evaluations

```bash
# Evaluate on test data
python -m src.evaluation.run --mode test --model xgboost --metric sharpe_ratio

# Full evaluation suite
python -m src.evaluation.full_suite --output ./evaluation_reports
```

### Portfolio Analysis

```python
# Example portfolio analysis script
from src.portfolio.analysis import PortfolioAnalyzer

analyzer = PortfolioAnalyzer(
    data_path="./data/portfolio_data.csv",
    risk_free_rate=0.02
)

results = analyzer.optimize_portfolio(
    method="mean_variance",
    constraints={"max_weight": 0.3}
)

print(f"Optimal weights: {results['weights']}")
print(f"Expected return: {results['return']:.2%}")
print(f"Portfolio risk: {results['risk']:.2%}")
```

## Future Enhancements

Planned features and improvements:

- Reinforcement learning strategies
- Real-time market data ingestion
- Advanced risk management modules
- Cloud deployment capabilities
- Automated feature engineering pipelines
- Enhanced portfolio optimization techniques

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This README provides comprehensive documentation of the project structure, setup instructions, and usage examples. The detailed directory structure ensures clarity about the project organization while providing practical guidance for developers.

# PRISM - Predictive Risk Intelligence System for Medicine

Unified refactored repository combining ML models, statistical analysis, and Flask API.
Funded by the Italian National Recovery and Resilience Plan (NRRP), M4C2, European Union –NextGenerationEU, through the Research Program “National Centre for HPC, Big Data and Quantum Computing”, Project CN00000013, Spoke 6, CUP B83C22002940006 (Cascade Funding Spoke 6).  
For any question and information please write to sscarpetta@unisa.it and agostino.aiezzo agostino.aiezzo@youbiquo.eu  (https://physlab.unicampus.it/prism/ )

## Structure

```
ml-prism/
├── src/
│   ├── models/          # PyTorch LSTM model definitions
│   ├── training/        # Training scripts (from main branch)
│   ├── stats/           # Statistical analysis (from statistics branch)
│   ├── api/             # Flask backend API (from Features branch)
│   └── utils/           # Shared utilities
├── models/              # Pre-trained model weights
├── datasets/            # MIMIC and SepsisExp data
├── docker/              # Docker configuration
└── tests/               # Test suite
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python -m src.api.app

# Or with Docker Compose
docker-compose up --build
```

## Branches Merged
- **main**: Core ML models (LSTM, TCN, RNN, Transformer) and training scripts
- **statistics**: Statistical analysis, explainability (SHAP, LIME), clustering
- **Features**: Flask backend API for clinical predictions

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Long-term forecasting experiments
python run_long-term_forecast.py

# Missing value imputation
python run_imputation.py

# Time series classification
python run_classification.py

# Probabilistic/interval forecasting
python run_probabilistic_forecast.py

# Few-shot transfer learning
python few_shot_training.py
```

No formal build/test/lint system - this is a research codebase.

## Architecture Overview

```
Entry Points (run_*.py)
    │
    ▼
Experiment Layer (exp/)
├── exp_basic.py          # Base class with model registry
├── exp_forecasting.py    # Training/testing for forecasting
├── exp_imputation.py     # Imputation with masking strategies
├── exp_classification.py # Classification workflows
└── exp_interval_forecasting.py  # Probabilistic outputs
    │
    ├───────────────┬──────────────────┐
    ▼               ▼                  ▼
Model Layer     Data Layer         Config Layer
(models/)       (data_provider/)   (configs/)
    │
    ▼
Utility Layer (utils/, layers/)
```

## Configuration System

Two-tier configuration:
1. **common_configs.py**: Base parameters (argparse) - model architecture, training settings
2. **Dataset configs** (e.g., electricity_configs.py): Override with dataset-specific values

Model-specific hyperparameters applied via `utils/setup_imputation.py`.

Key config fields:
- `task_name`: 'long_term_forecast' | 'imputation' | 'classification' | 'interval_forecast'
- `model`: 'DLinear' | 'Transformer' | 'TimesNet' | 'LLMformer' | etc.
- `seq_len`, `label_len`, `pred_len`: Sequence lengths
- `features`: 'M' (multivariate) | 'S' (univariate) | 'MS'

## Model Pattern

All models in `models/` extend `nn.Module` with task-branching in forward():
```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, ...):
    if self.task_name == 'long_term_forecast':
        return self.forecast(...)
    elif self.task_name == 'imputation':
        return self.imputation(...)
```

## Data Flow

1. `data_provider/data_factory.py` creates DataLoader via dataset class selection
2. Batch format: `(batch_x, batch_y, batch_x_mark, batch_y_mark, x_forecast)`
3. Shapes: `batch_x: [B, seq_len, features]`, output: `[B, pred_len, output_dim]`

## Key Files

- `exp/exp_basic.py:34-68` - Model registry dictionary
- `configs/common_configs.py` - All configurable parameters
- `utils/setup_imputation.py` - Model-specific hyperparameter setup
- `utils/tools.py` - EarlyStopping, learning rate adjustment
- `utils/metrics.py` - MAE, RMSE, MAPE evaluation

## Experiment Workflow

```python
from configs.electricity_configs import args
from utils.setup_imputation import model_hyparameter_setup
from exp.exp_forecasting import Exp_Forecast

args.model = 'TimesNet'
args.is_training = 1
args = model_hyparameter_setup(args)

exp = Exp_Forecast(args)
exp.train(setting)
res_df, metrics_df = exp.test(setting)
```

Results saved to `./results/{setting}/checkpoints/`.

## LLM-Integrated Models

Models like TimeLLM, LLMformer, AttLLM use pre-trained language models:
- Require prompt content from `dataset/prompt_bank/`
- LLM weights stored in `LLM/` directory
- Config: `args.llm_model` ('BERT', 'GPT2', etc.), `args.llm_dim`

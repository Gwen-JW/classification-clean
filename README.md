# classification-clean

## Installation

```bash
# clone project
git clone https://github.com/Gwen-JW/classification-clean.git
cd classification-clean
```

> **[Optional]**: Create virtual environment
```bash
virtualenv venv
source venv/bin/activate
```
> **[Required]**: Install all requirements
```bash
pip3 install -r requirements.txt
```

## Project structure

```
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── datamodule               <- Datamodule configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── src                    <- Source code
│   ├── datamodules              <- Datamodule scripts
│   ├── models                   <- Model scripts
│   ├── utils                    <- Utility scripts
│   │
│   ├── ucrformat.py             <- Download and preprocess UCR timeseries datasets
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── .gitignore                <- List of files ignored by git
├── .project-root             <- File for inferring the position of project root directory
├── requirements.txt          <- File for installing python dependencies
└── README.md
```

## How to run

### Example: Blink dataset

### 1. Download and prepare the time series dataset
```bash
python src/ucrformat.py
```
The dataset will be automatically downloaded and saved under `data/Blink`.


### 2. Train a model for time series classification
Train model with chosen experiment configuration from `configs/experiment/timeseries`. An example for Blink dataset is provided in [configs/experiment/timeseries/Blink.yaml](configs/experiment/timeseries/Blink.yaml).

```bash
python src/train.py experiment=timeseries/Blink.yaml
```
The trained results will be automatically saved to dir `logs/${task_name}/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}`.
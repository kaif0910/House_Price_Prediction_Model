# House Price Prediction

Simple PyTorch project for predicting house prices.

## Contents

- `train.py` — training script
- `predict.py` — inference / prediction script
- `saved_model.pth` — example trained model file
- `models/` — model definitions (`House_Price_Pred.py`)
- `utils/` — preprocessing helpers (`preprocess.py`)
- `requirements.txt` — Python dependencies

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1    # PowerShell
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Training

Run the training script to train a model (output saved to `saved_model.pth` by default):

```powershell
python train.py
```

Check `train.py` for configurable options such as epochs, batch size, and dataset paths.

## Inference / Prediction

Use the `predict.py` script to run inference with a saved model:

```powershell
python predict.py
```

Open `predict.py` for the expected input format and how predictions are produced.

## Project Structure

- `models/House_Price_Pred.py`: model architecture and related utilities.
- `utils/preprocess.py`: data preprocessing functions used by training and inference.

## Notes

- This repository expects dependencies listed in `requirements.txt`.
- If you encounter issues with PyTorch imports, verify your installed `torch` version and Python version compatibility.

## Contact

For questions or help, reply here or open an issue in the repository.

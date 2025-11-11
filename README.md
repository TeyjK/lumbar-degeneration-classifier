# Lumbar Degeneration Classifier

Deep learning system to classify lumbar spine degeneration severity from MRI scans (RSNA 2024).

## Overview

- Trained and tested deep learning models (**CNN, EfficientNetV2, ConvNeXt**) on 147k+ MRI slices representing ~2,000 patients.
- Multi-class classification across 3 spinal regions and 5 lumbar levels.
- Achieved **0.95 micro-average AUC** and **87% classification accuracy** on internal validation.

## Tech Stack

- **Backend**: Python, FastAPI, PyTorch
- **ML**: EfficientNetV2, ConvNeXt, Torchvision
- **Data**: DICOM via pydicom
- **Frontend**: HTML/CSS + vanilla JS (file upload, JSON results)

## How to run (local)

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn api.main:app --reload

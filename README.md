# Lumbar Degeneration Classifier

A production-ready deep learning system for classifying lumbar spine degeneration severity from MRI scans (RSNA 2024).  
Deployed as an interactive FastAPI web application for clinical exploration and model demonstration.

**Live Demo:** [https://lumbar-degeneration-classifier.onrender.com](https://lumbar-degeneration-classifier.onrender.com)

**Note:** This web app is hosted on Render’s free tier.  
The site may take up to 30 seconds to load if it has been inactive for a while, as the server spins back up.  
Once active, all features will function normally.

---

## Overview

This project implements a multi-class deep learning model to assess lumbar spine degeneration across three MRI sequences and multiple spinal regions.  
The trained models were deployed using FastAPI with a lightweight web interface designed for clinician usability.

**Highlights**
- Trained convolutional architectures (**CNN, EfficientNetV2, ConvNeXt**) on **147,000+ MRI slices** from ~2,000 patients.
- Classified degeneration severity (Normal/Mild, Moderate, Severe) for **three MRI sequences**: Sagittal T1, Sagittal T2/STIR, and Axial T2.
- Achieved **0.95 micro-average AUC** and **87% classification accuracy** on validation data.
- Deployed via **Render** as a FastAPI web application with real-time inference and visualization.

---

## Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Backend** | FastAPI, Uvicorn, Python |
| **Machine Learning** | PyTorch, Torchvision |
| **Data Handling** | NumPy, Pandas, PyDICOM |
| **Frontend** | HTML, CSS, Vanilla JavaScript |
| **Deployment** | Render (CPU environment) |

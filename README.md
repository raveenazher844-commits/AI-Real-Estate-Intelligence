
# EstateIQ — AI Real Estate Intelligence Platform

## Project Overview
EstateIQ is an educational AI-powered real estate application that
analyzes housing data, predicts property prices, recommends properties,
and provides a knowledge-based AI assistant.

## Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- FastAPI
- HTML, CSS
- Retrieval-Augmented Generation (RAG)
- Groq API (optional)
- GitHub

## Features
1. Exploratory Data Analysis and visualizations
2. Linear Regression, Decision Tree and Random Forest comparison
3. House price prediction
4. Budget and bedroom-based property recommendations
5. Knowledge retrieval and optional generative AI answers
6. FastAPI backend and interactive web interface

## Dataset
Download the House Prices: Advanced Regression Techniques dataset
from Kaggle and place train.csv inside the data folder.

Dataset:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

## Setup
Create and activate a virtual environment, then install dependencies:

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install -r requirements.txt

## Run
    python analysis.py
    python train_model.py
    python -m uvicorn app:app --reload

Open http://127.0.0.1:8000

API documentation: http://127.0.0.1:8000/docs

## Important Note
The model uses historical Ames, Iowa housing data. Predictions are
educational estimates and may not reflect current property market prices.

# # 🌍 Urban Heat Mitigation AI

## Problem Statement

Develop an AI-powered system to optimize urban heat mitigation and cooling strategies.

## Solution

This project predicts Land Surface Temperature (LST) using satellite-derived features from Google Earth Engine and an XGBoost regression model. The system classifies heat risk levels and recommends urban cooling strategies through an interactive Streamlit dashboard.

## Features

- Google Earth Engine data processing
- NDVI and NDBI computation
- XGBoost regression model
- Interactive temperature prediction
- Heat risk classification
- AI recommendations
- Interactive heat map

## Technologies

- Python
- Google Earth Engine
- XGBoost
- Streamlit
- PyDeck
- Pandas

## Model Performance

| Metric | Value |
|--------|-------|
| MAE | 1.06°C |
| RMSE | 1.41°C |
| R² | 0.645 |

## Project Structure

(Insert folder structure)

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py

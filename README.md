# DES SO<sub>2</sub> Absorption Prediction System

This is a web application for predicting SO<sub>2</sub> absorption capacity of different Deep Eutectic Solvents (DES) under various conditions. The system uses a deep learning model trained on experimental data to make predictions.

## Features

- Predict SO<sub>2</sub> absorption capacity for different DES combinations
- Visualize how absorption capacity changes with temperature, pressure, or water content
- Compare absorption capacity across different Hydrogen Bond Acceptors (HBA) or Hydrogen Bond Donors (HBD)
- Interactive web interface with real-time predictions and visualizations

## Requirements

- Python 3.7+
- TensorFlow 2.x
- FastAPI
- Pandas
- NumPy
- Plotly
- Joblib
- Uvicorn

## Installation

1. Clone this repository

```bash
git clone https://github.com/nonfakelz/des-so2-absorption.git
cd des-so2-absorption
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Make sure you have the following files in your project directory:
   - `best_model.keras`: The trained TensorFlow model
   - `x_scaler.pkl`: Scaler for input features
   - `y_scaler.pkl`: Scaler for output values

## Usage

1. Start the web server:

```bash
python main.py
```

2. Open your browser and navigate to `http://localhost:8000`

3. Use the web interface to:
   - Select HBA and HBD types
   - Set DES ratio, temperature, pressure, and water content
   - Get predictions for specific conditions
   - Generate plots showing how absorption capacity changes with different variables

## API Endpoints

- `GET /`: Home page with the web interface
- `POST /predict`: Get a single prediction for specific conditions
- `POST /predict_range`: Get predictions for a range of values of a selected variable
- `POST /predict_category`: Compare predictions across different HBA or HBD types

## Model Information

The prediction model is a neural network trained on experimental data of SO<sub>2</sub> absorption in various DES systems. The model takes the following inputs:

- HBA type (one-hot encoded)
- HBD type (one-hot encoded)
- DES ratio
- Temperature (K)
- Pressure (kPa)
- Water content (%)

And outputs the SO<sub>2</sub> absorption capacity in g/g (grams of SO<sub>2</sub> per gram of DES).

## License

This project is licensed under the [MIT](https://opensource.org/license/mit/) License.
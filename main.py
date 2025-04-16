import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="DES SO<sub>2</sub> Absorption Prediction System", 
              description="Predict SO<sub>2</sub> absorption capacity of different DES under various conditions")

templates = Jinja2Templates(directory="templates")

os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

try:
    model = tf.keras.models.load_model('best_model.keras', compile=False)
    
    x_scaler = joblib.load('x_scaler.pkl')
    y_scaler = joblib.load('y_scaler.pkl')
    
    hba_list = ['[Emim]Cl', '[N4444]Cl', '[P4444]Cl', '[Bmim]Cl', '[Ch]Cl', 'ACC', '[Bmim]Br', 
                '[PPZ]Br', '[Hmim]Cl', 'Bet', 'L-car', 'CPL', 'Mat', '[Amim]Cl', '[MImH]Cl', 
                '[BImH]Cl', '[EImH]Cl', '[Emim]Br', '[EimH]Cl', '[EimH][MOA]', '[EimH][TFA]', 
                '[TEA]Cl']
    
    hbd_list = ['DCDA', 'Im', 'Ben-Im', 'Pyr', 'Tetz', '4-Mim', 'NFM', '[Epy]Br', 'EG', 'MA', 
                'Urea', 'Thiourea', 'Triz', 'IMD', 'DMU', 'SN', 'Gly', 'Tz', 'PYD', 'PID', 
                'CLAA', 'SUIM', 'NHS', 'MU', 'SAA', 'AA', 'Benzoicacid', 'MFA', 'EFA', 'MAA', 
                '2-NH2Py', '3-NH2Py', '3-OHPy', '2-Mim', 'GC', 'CL', '[Epy]Cl', '4-NH2Py', 
                '4-OHPy', '1, 3-PDO', 'TG', '2-OHPy']
    
    temp_range = (293.0, 353.5)
    pressure_range = (0.2, 127.3)
    water_range = (0.0, 20.0)
    
    try:
        if hasattr(x_scaler, 'feature_names_in_'):
            feature_columns = x_scaler.feature_names_in_.tolist()
            logger.info(f"Feature names loaded from x_scaler: {len(feature_columns)} features")
        else:
            logger.warning("No feature_names_in_ attribute in x_scaler, will create feature name list")
            raise AttributeError("No feature_names_in_ attribute")
    except (AttributeError, FileNotFoundError):
        logger.warning("Unable to load feature name list, creating new feature name list")
    
    logger.info(f"Model and data loaded successfully")
    logger.info(f"Number of HBA types: {len(hba_list)}")
    logger.info(f"Number of HBD types: {len(hbd_list)}")
    logger.info(f"Temperature range: {temp_range}")
    logger.info(f"Pressure range: {pressure_range}")
    logger.info(f"Water content range: {water_range}")
    
except Exception as e:
    logger.error(f"Failed to load model or data: {e}")
    raise

def create_feature_columns():
    """Create feature column names for model input"""
    numerical_features = ['DES_ratio', 'T(K)', 'P(kPa)', 'w(H2O)%']
    
    hba_columns = [f'HBA_{hba}' for hba in hba_list]
    hbd_columns = [f'HBD_{hbd}' for hbd in hbd_list]
    
    return numerical_features + hba_columns + hbd_columns

if 'feature_columns' not in globals():
    feature_columns = create_feature_columns()
    logger.warning("Using newly created feature column list, may be inconsistent with training")

def predict_absorption(hba: str, hbd: str, des_ratio: float, temperature: float, 
                       pressure: float, water_content: float) -> float:
    """Predict SO2 absorption capacity using the loaded model"""
    try:
        input_data = pd.DataFrame({
            'DES_ratio': [des_ratio],
            'T(K)': [temperature],
            'P(kPa)': [pressure],
            'w(H2O)%': [water_content]
        })
        
        for h in hba_list:
            input_data[f'HBA_{h}'] = 1 if h == hba else 0
            
        for h in hbd_list:
            input_data[f'HBD_{h}'] = 1 if h == hbd else 0
        
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)
        
        input_scaled = x_scaler.transform(input_data)
        
        prediction_scaled = model.predict(input_scaled)
        
        prediction = y_scaler.inverse_transform(prediction_scaled)
        
        return float(prediction[0][0])
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

def generate_prediction_data(hba: str, hbd: str, des_ratio: float, 
                           temperature: Optional[float], pressure: Optional[float], 
                           water_content: Optional[float], 
                           disabled_var: str, var_range: List[float]) -> Dict[str, Any]:
    """Generate prediction data for a range of variable values"""
    predictions = []
    
    for val in var_range:
        temp = val if disabled_var == 'temperature' else temperature
        press = val if disabled_var == 'pressure' else pressure
        water = val if disabled_var == 'water_content' else water_content
        
        pred = predict_absorption(hba, hbd, des_ratio, temp, press, water)
        predictions.append(pred)
    
    return {
        'variable': var_range,
        'predictions': predictions
    }

def generate_categorical_prediction_data(fixed_category: str, fixed_value: str, 
                                       variable_category: str, des_ratio: float,
                                       temperature: float, pressure: float, 
                                       water_content: float) -> Dict[str, Any]:
    """Generate prediction data for different HBA or HBD types"""
    if variable_category == 'HBA':
        categories = hba_list
    else:  # HBD
        categories = hbd_list
    
    predictions = []
    
    for cat in categories:
        hba = cat if variable_category == 'HBA' else fixed_value
        hbd = fixed_value if variable_category == 'HBA' else cat
        
        pred = predict_absorption(hba, hbd, des_ratio, temperature, pressure, water_content)
        predictions.append(pred)
    
    return {
        'categories': categories,
        'predictions': predictions
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "hba_list": hba_list, "hbd_list": hbd_list,
         "temp_min": temp_range[0], "temp_max": temp_range[1],
         "pressure_min": pressure_range[0], "pressure_max": pressure_range[1],
         "water_min": water_range[0], "water_max": water_range[1]}
    )

@app.post("/predict")
async def predict(request: Request):
    """Single prediction API"""
    form_data = await request.form()
    
    try:
        hba = form_data.get("hba")
        hbd = form_data.get("hbd")
        des_ratio = float(form_data.get("des_ratio"))
        temperature = float(form_data.get("temperature"))
        pressure = float(form_data.get("pressure"))
        water_content = float(form_data.get("water_content"))
        
        prediction = predict_absorption(hba, hbd, des_ratio, temperature, pressure, water_content)
        
        return JSONResponse({
            "prediction": prediction,
            "units": "g/g"
        })
    
    except Exception as e:
        logger.error(f"Prediction API error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_range")
async def predict_range(request: Request):
    """Range prediction API"""
    form_data = await request.form()
    
    try:
        hba = form_data.get("hba")
        hbd = form_data.get("hbd")
        des_ratio = float(form_data.get("des_ratio"))
        
        disabled_var = form_data.get("disabled_var")
        
        temperature = float(form_data.get("temperature")) if disabled_var != "temperature" else None
        pressure = float(form_data.get("pressure")) if disabled_var != "pressure" else None
        water_content = float(form_data.get("water_content")) if disabled_var != "water_content" else None
        
        if disabled_var == "temperature":
            var_min = float(form_data.get("temp_min"))
            var_max = float(form_data.get("temp_max"))
            var_step = (var_max - var_min) / 20  # Generate 20 points
            var_range = np.arange(var_min, var_max + var_step, var_step).tolist()
            x_label = "Temperature (K)"
        elif disabled_var == "pressure":
            var_min = float(form_data.get("pressure_min"))
            var_max = float(form_data.get("pressure_max"))
            var_step = (var_max - var_min) / 20
            var_range = np.arange(var_min, var_max + var_step, var_step).tolist()
            x_label = "Pressure (kPa)"
        elif disabled_var == "water_content":
            var_min = float(form_data.get("water_min"))
            var_max = float(form_data.get("water_max"))
            var_step = (var_max - var_min) / 20
            var_range = np.arange(var_min, var_max + var_step, var_step).tolist()
            x_label = "Water Content (%)"
        else:
            return JSONResponse({"error": "No variable specified"})
        
        prediction_data = generate_prediction_data(
            hba, hbd, des_ratio, temperature, pressure, water_content, 
            disabled_var, var_range
        )
        
        fig = px.line(
            x=prediction_data['variable'], 
            y=prediction_data['predictions'],
            labels={"x": x_label, "y": "SO2 Absorption Capacity (g/g)"},
            title=f"SO2 Absorption Capacity vs {x_label}"
        )
        
        return JSONResponse({
            "plot": fig.to_json(),
            "data": {
                "x": prediction_data['variable'],
                "y": prediction_data['predictions'],
                "x_label": x_label,
                "y_label": "SO2 Absorption Capacity (g/g)"
            }
        })
    
    except Exception as e:
        logger.error(f"Range prediction API error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_category")
async def predict_category(request: Request):
    """HBA/HBD category prediction API"""
    form_data = await request.form()
    
    try:
        fixed_category = form_data.get("fixed_category")  # 'HBA' or 'HBD'
        fixed_value = form_data.get("fixed_value")
        
        variable_category = "HBD" if fixed_category == "HBA" else "HBA"
        
        des_ratio = float(form_data.get("des_ratio"))
        temperature = float(form_data.get("temperature"))
        pressure = float(form_data.get("pressure"))
        water_content = float(form_data.get("water_content"))
        
        prediction_data = generate_categorical_prediction_data(
            fixed_category, fixed_value, variable_category, 
            des_ratio, temperature, pressure, water_content
        )
        
        fig = px.bar(
            x=prediction_data['categories'],
            y=prediction_data['predictions'],
            labels={"x": f"{variable_category} Type", "y": "SO2 Absorption Capacity (g/g)"},
            title=f"SO2 Absorption Capacity Comparison for Different {variable_category} Types"
        )
        
        return JSONResponse({
            "plot": fig.to_json(),
            "data": {
                "x": prediction_data['categories'],
                "y": prediction_data['predictions'],
                "x_label": f"{variable_category} Type",
                "y_label": "SO2 Absorption Capacity (g/g)"
            }
        })
    
    except Exception as e:
        logger.error(f"Category prediction API error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
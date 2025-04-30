from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import joblib
import re

# --- Inisialisasi FastAPI
app = FastAPI()

# --- Load Model, Scalers, and Trained Columns
sc_X = joblib.load('./model/scaler_X.pkl')
sc_y = joblib.load('./model/scaler_y.pkl')
svr_model = joblib.load('./model/house_rent_model.pkl')
trained_columns = joblib.load('./model/columns.pkl')

# --- Ekstrak Floor
def extract_floor_data(floor_str):
    try:
        parts = floor_str.split(' out of ')
        floor = parts[0]
        total = parts[1]

        if floor.strip().lower() == 'ground':
            floor_num = 0
        else:
            floor_num = int(floor)
        total_floor = int(total)
    except:
        floor_num = np.nan
        total_floor = np.nan
    return floor_num, total_floor

# --- Model untuk input
class RentInput(BaseModel):
    bhk: int = Field(..., gt=0, description="Jumlah kamar harus lebih dari 0")
    size: int = Field(..., gt=0, description="Ukuran properti dalam sqft")
    bathroom: int = Field(..., ge=0, description="Jumlah kamar mandi")
    floor: str = Field(..., example="2 out of 5")
    area_type: str
    area_locality: str
    city: str
    furnishing_status: str
    tenant_preferred: str
    point_of_contact: str

    @validator('floor')
    def validate_floor_format(cls, v):
        if not re.match(r"^(Ground|\d+)\s+out\s+of\s+\d+$", v.strip(), re.IGNORECASE):
            raise ValueError("Format Floor harus seperti 'Ground out of 5' atau '2 out of 10'")
        return v

    @validator('area_type', 'area_locality', 'city', 'furnishing_status', 'tenant_preferred', 'point_of_contact')
    def no_empty_strings(cls, v):
        if not v.strip():
            raise ValueError("Field tidak boleh kosong")
        return v

# --- Endpoint prediksi
@app.post("/predict")
def predict_rent(input_data: RentInput):
    try:
        floor_num, total_floor = extract_floor_data(input_data.floor)

        df_input = pd.DataFrame([{
            'BHK': input_data.bhk,
            'Size': input_data.size,
            'Bathroom': input_data.bathroom,
            'floor_num': floor_num,
            'total_floor': total_floor,
            'Area Type': input_data.area_type,
            'Area Locality': input_data.area_locality,
            'City': input_data.city,
            'Furnishing Status': input_data.furnishing_status,
            'Tenant Preferred': input_data.tenant_preferred,
            'Point of Contact': input_data.point_of_contact
        }])

        df_input_encoded = pd.get_dummies(df_input)

        missing_cols = [col for col in trained_columns if col not in df_input_encoded.columns]
        missing_df = pd.DataFrame(0, index=df_input_encoded.index, columns=missing_cols)
        df_input_encoded = pd.concat([df_input_encoded, missing_df], axis=1)

        df_input_encoded = df_input_encoded[trained_columns]

        X_scaled = sc_X.transform(df_input_encoded)

        y_scaled_pred = svr_model.predict(X_scaled)
        y_pred = sc_y.inverse_transform(y_scaled_pred.reshape(-1, 1))

        return {"predicted_rent": int(y_pred[0][0])}

    except Exception as e:
        return {"detail": f"Prediction error: {str(e)}"}

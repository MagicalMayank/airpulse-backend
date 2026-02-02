from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import pickle
import pandas as pd
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI & ML INITIALIZATION ---
# DeepSeek Setup
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Load Model Resources
MODEL_PATH = "model/airpulse_unified_model.pkl"
ENCODER_PATH = "model/city_encoder.pkl"

model = None
city_encoder = None

def load_resources():
    global model, city_encoder
    if model is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(ENCODER_PATH, "rb") as f:
                city_encoder = pickle.load(f)
            print("✅ Unified Model & City Encoder Loaded")
        except Exception as e:
            print(f"❌ Load Error: {e}")

@app.on_event("startup")
async def startup():
    load_resources()

# --- ROUTES ---

@app.get("/")
@app.head("/")
async def health_check():
    return {"status": "online", "engine": "DeepSeek + XGBoost"}

@app.get("/policy-ai")
async def policy_ai(query: str, city: str):
    """Translates text policies to numerical slider values using DeepSeek"""
    system_prompt = f"""
    You are an Air Quality Policy Expert for AirPulse 2.0. 
    Translate the user's policy into reduction percentages (0-100) for:
    1. traffic_reduction
    2. dust_reduction

    Context:
    - City: {city}
    - Training Data: 2015-2024 Indian City AQI
    - Logic: Traffic reduces NO2; Dust control reduces PM10
    - Specifics: If user mentions GRAP or Odd-Even, follow standard Delhi protocols.

    Return ONLY a JSON object: {{"traffic": float, "dust": float, "reasoning": "string"}}
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            response_format={'type': 'json_object'}
        )
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

@app.get("/simulate")
async def simulate(city: str, baseline_aqi: float, traffic_reduction: float, dust_reduction: float):
    if model is None: load_resources()
    try:
        # 1. City Code
        try:
            city_code = city_encoder.transform([city])[0]
        except:
            city_code = 0
        
        # 2. Proxy Transformation
        sim_traffic = 50.0 * (1 - traffic_reduction/100)
        sim_dust = 50.0 * (1 - dust_reduction/100)
        
        # 3. Predict
        input_data = pd.DataFrame(
            [[city_code, baseline_aqi, sim_traffic, sim_dust]], 
            columns=['City_Code', 'AQI_lag_1', 'traffic_proxy', 'dust_proxy']
        )
        prediction = float(model.predict(input_data)[0])
        
        # 4. Scaling Logic
        raw_reduction = baseline_aqi - prediction
        final_reduction = max(5, raw_reduction) if (traffic_reduction + dust_reduction) > 0 else 0
        projected_aqi = max(10, round(baseline_aqi - final_reduction))

        return {
            "city": city,
            "original_aqi": baseline_aqi,
            "projected_aqi": projected_aqi,
            "total_reduction": round(final_reduction, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

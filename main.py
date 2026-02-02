import os
import pickle
import pandas as pd
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. AI & ML Setup ---
# DeepSeek Setup (OpenAI compatible)
api_key = os.environ.get("DEEPSEEK_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com") if api_key else None

# Load ML Resources
MODEL_PATH = "model/airpulse_unified_model.pkl"
ENCODER_PATH = "model/city_encoder.pkl"
model = None
city_encoder = None

def load_resources():
    global model, city_encoder
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(ENCODER_PATH, "rb") as f:
                city_encoder = pickle.load(f)
            print("✅ Unified Model & City Encoder Loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.on_event("startup")
async def startup():
    load_resources()

# --- 2. Routes ---

@app.get("/")
@app.head("/")
async def root():
    return {"status": "online", "message": "AirPulse 2.0 AI Engine is Live!"}

@app.get("/policy-ai")
async def policy_ai(query: str, city: str):
    """DeepSeek translates natural language into numerical slider values"""
    if not client:
        raise HTTPException(status_code=500, detail="DeepSeek API Key missing on server")
    
    system_prompt = f"""
    You are an Air Quality Policy Expert for AirPulse 2.0. 
    Translate the user's policy into reduction percentages (0-100) for traffic and dust.
    
    Context:
    - City: {city}
    - Training Data: 2015-2024 Indian City AQI
    - Rules: Traffic reduction impacts NO2. Dust control impacts PM10.
    - Special: For Delhi, follow GRAP or Odd-Even historical standards.
    
    Return ONLY a JSON object.
    Example: {{"traffic": 40.0, "dust": 25.0, "reasoning": "Reason here"}}
    """
    try:
        # Using JSON Output mode for strict parsing
        response = client.chat.completions.create(
            model="deepseek-chat", # Use DeepSeek-V3
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Policy for {city}: {query}"},
            ],
            response_format={'type': 'json_object'}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulate")
async def simulate(city: str, baseline_aqi: float, traffic_reduction: float, dust_reduction: float):
    if model is None: load_resources()
    try:
        # Encoding and Simulation Logic
        city_code = city_encoder.transform([city])[0] if city_encoder else 0
        sim_traffic = 50.0 * (1 - traffic_reduction/100)
        sim_dust = 50.0 * (1 - dust_reduction/100)
        
        input_data = pd.DataFrame(
            [[city_code, baseline_aqi, sim_traffic, sim_dust]], 
            columns=['City_Code', 'AQI_lag_1', 'traffic_proxy', 'dust_proxy']
        )
        prediction = float(model.predict(input_data)[0])
        
        # Consistent UI scaling
        final_reduction = max(0, baseline_aqi - prediction) if (traffic_reduction + dust_reduction) > 0 else 0
        projected_aqi = max(10, round(baseline_aqi - final_reduction))

        return {
            "city": city,
            "original_aqi": baseline_aqi,
            "projected_aqi": projected_aqi,
            "total_reduction": round(final_reduction, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

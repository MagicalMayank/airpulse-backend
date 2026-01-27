from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import uvicorn
import os

# Load environment variables from .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

app = FastAPI(title="AirPulse Policy Simulator API")

# 1. CORS enable karein (Frontend integration ke liye zaroori hai)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Model Load karein
MODEL_PATH = "model/policy_simulator_xgb.pkl"
model = None
explainer = None

def load_model():
    """Lazy load model and SHAP explainer"""
    global model, explainer
    if model is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            print("✅ Model loaded successfully")
            
            # Try to initialize SHAP explainer
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                print("✅ SHAP explainer initialized")
            except Exception as e:
                print(f"⚠️ SHAP explainer failed (will use fallback): {e}")
                explainer = None
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise e
    return model, explainer

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"⚠️ Model not loaded at startup: {e}")

@app.get("/")
def home():
    return {"message": "AirPulse Backend is Live!"}

@app.get("/simulate")
async def simulate(
    ward: str,
    baseline_aqi: float,
    traffic: float,   # Frontend se short name aayega
    dust: float,      # Frontend se short name aayega
    biomass: int,     # Frontend se short name aayega
    weather: int      # Frontend se short name aayega
):
    try:
        # STEP: Translator logic (Mapping short names to Model's long names)
        # Model ko wahi column names chahiye jo training ke waqt the
        input_df = pd.DataFrame(
            [[baseline_aqi, traffic, dust, biomass, weather]], 
            columns=[
                'baseline_aqi', 
                'traffic_diversion',  # Model ye dhoond raha hai
                'dust_control',      # Model ye dhoond raha hai
                'biomass_burning',    # Model ye dhoond raha hai
                'weather_assistance'  # Model ye dhoond raha hai
            ]
        )

        # Prediction logic
        reduction_pred = model.predict(input_df)[0]
        final_aqi = max(0, baseline_aqi - reduction_pred)

        # SHAP calculation
        shap_values = explainer.shap_values(input_df)

        return {
            "status": "success",
            "ward": ward,
            "projected_aqi": round(final_aqi),
            "impact_breakdown": {
                "traffic": round(abs(float(shap_values[0][1])), 2),
                "dust": round(abs(float(shap_values[0][2])), 2),
                "biomass": round(abs(float(shap_values[0][3])), 2),
                "weather": round(abs(float(shap_values[0][4])), 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
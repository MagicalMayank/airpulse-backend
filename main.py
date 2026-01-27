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
    ward: str,           # Ward ID (e.g., "123" or "Ward 45")
    baseline_aqi: float, # Current AQI from frontend 
    traffic: float,      # Traffic diversion % (0-30)
    dust: float,         # Dust control % (0-40)
    biomass: int,        # Biomass burning enforcement (0 or 1)
    weather: int         # Weather assistance (0=none, 1=moderate, 2=strong)
):
    """
    Run policy simulation for a specific ward.
    
    The frontend sends the ward's current AQI (baseline_aqi) directly,
    so we don't need to fetch from WAQI API.
    """
    try:
        # Load model if not already loaded
        model_instance, shap_explainer = load_model()
        
        # STEP A: Prepare Data for Prediction
        # Sequence: baseline_aqi, traffic_diversion, dust_control, biomass_burning, weather_assistance
        input_df = pd.DataFrame(
            [[baseline_aqi, traffic, dust, biomass, weather]], 
            columns=['baseline_aqi', 'traffic_diversion', 'dust_control', 'biomass_burning', 'weather_assistance']
        )

        # STEP B: Model Prediction
        reduction_pred = float(model_instance.predict(input_df)[0])
        final_aqi = max(0, baseline_aqi - reduction_pred)

        # STEP C: XAI (SHAP) - Policy Impact Breakdown
        impact = {
            "traffic_impact": 0.0,
            "dust_impact": 0.0,
            "biomass_impact": 0.0,
            "weather_impact": 0.0
        }
        
        if shap_explainer is not None:
            try:
                shap_values = shap_explainer.shap_values(input_df)
                # SHAP values array: [baseline, traffic, dust, biomass, weather]
                impact = {
                    "traffic_impact": round(abs(float(shap_values[0][1])), 2),
                    "dust_impact": round(abs(float(shap_values[0][2])), 2),
                    "biomass_impact": round(abs(float(shap_values[0][3])), 2),
                    "weather_impact": round(abs(float(shap_values[0][4])), 2)
                }
            except Exception as e:
                print(f"⚠️ SHAP calculation failed: {e}")
                # Use heuristic fallback for impact breakdown
                total_reduction = reduction_pred if reduction_pred > 0 else 1
                impact = {
                    "traffic_impact": round((traffic / 30) * total_reduction * 0.35, 2),
                    "dust_impact": round((dust / 40) * total_reduction * 0.25, 2),
                    "biomass_impact": round(biomass * total_reduction * 0.25, 2),
                    "weather_impact": round((weather / 2) * total_reduction * 0.15, 2)
                }
        else:
            # Heuristic fallback when SHAP is not available
            total_reduction = reduction_pred if reduction_pred > 0 else 1
            impact = {
                "traffic_impact": round((traffic / 30) * total_reduction * 0.35, 2),
                "dust_impact": round((dust / 40) * total_reduction * 0.25, 2),
                "biomass_impact": round(biomass * total_reduction * 0.25, 2),
                "weather_impact": round((weather / 2) * total_reduction * 0.15, 2)
            }

        return {
            "status": "success",
            "ward": ward,
            "baseline_aqi": baseline_aqi,
            "projected_aqi": round(final_aqi),
            "total_reduction": round(reduction_pred),
            "impact_breakdown": impact,
            "confidence_score": 0.85  # Static for demo
        }

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found. Please ensure policy_simulator_xgb.pkl exists in model/ folder.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
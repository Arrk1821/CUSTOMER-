from typing import Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as uvicorn_run

# -------------------------
# ✅ Correct Imports
# -------------------------
# Utility to read YAML config
# Remove SCHEMA_FILE_PATH if not used
from src.utils.main_utils import read_yaml_file


# Pipeline modules
from src.pipeline.prediction_pipeline import ChurnData, ChurnPredictor
from src.pipeline.training_pipeline import TrainPipeline


# -------------------------
# Load configuration
# -------------------------
config = read_yaml_file("config.yaml")
APP_HOST = config.get("app_host", "127.0.0.1")
APP_PORT = config.get("app_port", 8000)


# -------------------------
# FastAPI App Setup
# -------------------------
app = FastAPI(title="ChurnCast")

# Static files + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Helper Class for Form Data
# -------------------------
class DataForm:
    def __init__(self, request: Request):
        self.request = request
        self.fields = [
            "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
            "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
            "Complain", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
            "DaySinceLastOrder", "CashbackAmount", "Gender", "PreferedOrderCat",
            "MaritalStatus", "PreferredLoginDevice", "PreferredPaymentMode"
        ]

    async def get_data(self):
        form = await self.request.form()
        return {key: form.get(key) for key in self.fields}


# -------------------------
# Routes
# -------------------------
@app.get("/", tags=["UI"])
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "context": "Rendering"}
    )


@app.get("/train", tags=["Model"])
async def train_model(request: Request):
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        message = "✅ Model training completed successfully!"
    except Exception as e:
        message = f"❌ Training failed: {e}"
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "context": message}
    )


@app.post("/", tags=["Prediction"])
async def predict(request: Request):
    try:
        form = DataForm(request)
        data = await form.get_data()

        churn_data = ChurnData(
            Tenure=float(data["Tenure"]),
            CityTier=int(data["CityTier"]),
            WarehouseToHome=float(data["WarehouseToHome"]),
            HourSpendOnApp=float(data["HourSpendOnApp"]),
            NumberOfDeviceRegistered=int(data["NumberOfDeviceRegistered"]),
            SatisfactionScore=int(data["SatisfactionScore"]),
            NumberOfAddress=int(data["NumberOfAddress"]),
            Complain=int(data["Complain"]),
            OrderAmountHikeFromlastYear=float(data["OrderAmountHikeFromlastYear"]),
            CouponUsed=float(data["CouponUsed"]),
            OrderCount=float(data["OrderCount"]),
            DaySinceLastOrder=float(data["DaySinceLastOrder"]),
            CashbackAmount=float(data["CashbackAmount"]),
            Gender=data["Gender"],
            PreferedOrderCat=data["PreferedOrderCat"],
            MaritalStatus=data["MaritalStatus"],
            PreferredLoginDevice=data["PreferredLoginDevice"],
            PreferredPaymentMode=data["PreferredPaymentMode"],
        )

        churn_df = churn_data.get_churn_input_data_frame()
        predictor = ChurnPredictor()
        prediction = predictor.predict(dataframe=churn_df)

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": f"Prediction: {prediction}"}
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": f"❌ Error during prediction: {e}"}
        )


# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    uvicorn_run("app:app", host=APP_HOST, port=APP_PORT, reload=True)

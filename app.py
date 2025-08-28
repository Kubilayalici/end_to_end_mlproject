from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object
import os


app = FastAPI(title="Student Final Grade Predictor", version="1.0.0")

# Ensure static directory exists, then mount
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def _model_name_or_unknown() -> str:
    try:
        model = load_object("artifacts/model.pkl")
        return type(model).__name__
    except Exception:
        return "Unknown"

@app.get("/health", response_class=HTMLResponse)
async def health() -> str:
    return "OK"


@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "results": None, "model_name": _model_name_or_unknown()})


@app.post("/predict", response_class=HTMLResponse, name="predict")
async def post_predict(
    request: Request,
    school: str = Form(...),
    sex: str = Form(...),
    age: int = Form(...),
    address: str = Form(...),
    famsize: str = Form(...),
    Pstatus: str = Form(...),
    Medu: int = Form(...),
    Fedu: int = Form(...),
    Mjob: str = Form(...),
    Fjob: str = Form(...),
    reason: str = Form(...),
    guardian: str = Form(...),
    traveltime: int = Form(...),
    studytime: int = Form(...),
    failures: int = Form(...),
    schoolsup: str = Form(...),
    famsup: str = Form(...),
    paid: str = Form(...),
    activities: str = Form(...),
    nursery: str = Form(...),
    higher: str = Form(...),
    internet: str = Form(...),
    romantic: str = Form(...),
    famrel: int = Form(...),
    freetime: int = Form(...),
    goout: int = Form(...),
    Dalc: int = Form(...),
    Walc: int = Form(...),
    health: int = Form(...),
    absences: int = Form(...),
    G1: int = Form(...),
    G2: int = Form(...),
):
    data = CustomData(
        school=school,
        sex=sex,
        age=age,
        address=address,
        famsize=famsize,
        Pstatus=Pstatus,
        Medu=Medu,
        Fedu=Fedu,
        Mjob=Mjob,
        Fjob=Fjob,
        reason=reason,
        guardian=guardian,
        traveltime=traveltime,
        studytime=studytime,
        failures=failures,
        schoolsup=schoolsup,
        famsup=famsup,
        paid=paid,
        activities=activities,
        nursery=nursery,
        higher=higher,
        internet=internet,
        romantic=romantic,
        famrel=famrel,
        freetime=freetime,
        goout=goout,
        Dalc=Dalc,
        Walc=Walc,
        health=health,
        absences=absences,
        G1=G1,
        G2=G2,
    )

    pred_df = data.to_dataframe()
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return templates.TemplateResponse("home.html", {"request": request, "results": float(results[0]), "model_name": _model_name_or_unknown()})


# For local run: uvicorn app:app --reload

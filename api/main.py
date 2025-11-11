from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from io import BytesIO
from src.inference import predict

app = FastAPI(title="Lumbar Degeneration Classifier")

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
templates = Jinja2Templates(directory="frontend")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def classify_image(
    file: UploadFile = File(...),
    model_type: str = Form("Sagittal T1")
):
    data = BytesIO(await file.read())
    result = predict(data, model_type)
    return {"probabilities": result}

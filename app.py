
import os
import re
import joblib
import requests
import pandas as pd

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

app = FastAPI(title="AI Real Estate Intelligence Platform")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/plots", StaticFiles(directory="plots"), name="plots")
templates = Jinja2Templates(directory="templates")

DATA_PATH = "data/train.csv"
MODEL_PATH = "models/price_model.pkl"
KNOWLEDGE_PATH = "knowledge.txt"

def get_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def get_properties():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    df = get_properties()
    total_properties = len(df)
    average_price = 0

    if not df.empty and "SalePrice" in df.columns:
        average_price = int(df["SalePrice"].mean())

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "total_properties": total_properties,
            "average_price": average_price,
            "prediction": None,
            "recommendations": [],
            "error": None,
            "chat_question": None,
            "chat_answer": None,
        },
    )

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    area: float = Form(...),
    quality: int = Form(...),
    year_built: int = Form(...),
    basement: float = Form(...),
    garage: int = Form(...),
    bathrooms: int = Form(...),
    bedrooms: int = Form(...),
    lot_area: float = Form(...),
):
    bundle = get_bundle()
    df = get_properties()
    prediction = None
    error = None

    try:
        if bundle is None:
            raise ValueError("Model missing. Run python train_model.py first.")

        if (
            area <= 0 or basement < 0 or lot_area <= 0
            or not 1 <= quality <= 10
            or not 1800 <= year_built <= 2026
            or not 0 <= garage <= 5
            or not 0 <= bathrooms <= 5
            or not 0 <= bedrooms <= 10
        ):
            raise ValueError("Please enter valid property details.")

        features = bundle["features"]
        row = pd.DataFrame([{
            "GrLivArea": area,
            "OverallQual": quality,
            "YearBuilt": year_built,
            "TotalBsmtSF": basement,
            "GarageCars": garage,
            "FullBath": bathrooms,
            "BedroomAbvGr": bedrooms,
            "LotArea": lot_area,
        }])[features]

        prediction = round(float(bundle["model"].predict(row)[0]), 2)

    except Exception as exc:
        error = str(exc)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "total_properties": len(df),
            "average_price": int(df["SalePrice"].mean())
                if not df.empty and "SalePrice" in df.columns else 0,
            "prediction": prediction,
            "recommendations": [],
            "error": error,
            "chat_question": None,
            "chat_answer": None,
        },
    )

@app.post("/recommend", response_class=HTMLResponse)
def recommend(
    request: Request,
    budget: float = Form(...),
    bedrooms: int = Form(...),
    location: str = Form(""),
):
    df = get_properties()
    recommendations = []
    error = None

    try:
        if df.empty:
            raise ValueError("Dataset missing. Add data/train.csv first.")

        if budget <= 0 or bedrooms < 0:
            raise ValueError("Enter a positive budget and valid bedrooms.")

        filtered = df[
            (df["SalePrice"] <= budget)
            & (df["BedroomAbvGr"] >= bedrooms)
        ].copy()

        if location.strip():
            if "Neighborhood" in filtered.columns:
                filtered = filtered[
                    filtered["Neighborhood"].astype(str).str.contains(
                        re.escape(location.strip()),
                        case=False,
                        na=False,
                    )
                ]
            else:
                raise ValueError("Location information is unavailable.")

        filtered = filtered.sort_values("SalePrice").head(6)

        for _, item in filtered.iterrows():
            recommendations.append({
                "price": int(item["SalePrice"]),
                "bedrooms": int(item["BedroomAbvGr"]),
                "area": int(item["GrLivArea"]),
                "quality": int(item["OverallQual"]),
                "location": str(item.get("Neighborhood", "Not available")),
                "year": int(item["YearBuilt"]),
            })

    except Exception as exc:
        error = str(exc)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "total_properties": len(df),
            "average_price": int(df["SalePrice"].mean())
                if not df.empty and "SalePrice" in df.columns else 0,
            "prediction": None,
            "recommendations": recommendations,
            "error": error,
            "chat_question": None,
            "chat_answer": None,
        },
    )

def retrieve_knowledge(question: str, top_k: int = 3) -> str:
    if not os.path.exists(KNOWLEDGE_PATH):
        return "Knowledge file is not available."

    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = [
        part.strip()
        for part in re.split(r"\n\s*\n", text)
        if part.strip()
    ]

    words = set(re.findall(r"\w+", question.lower()))
    scored = []

    for chunk in chunks:
        chunk_words = set(re.findall(r"\w+", chunk.lower()))
        score = len(words & chunk_words)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)

    if not scored:
        return text[:2500]

    return "\n\n".join(chunk for _, chunk in scored[:top_k])

def answer_question(question: str) -> str:
    context = retrieve_knowledge(question)
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        return (
            "AI knowledge assistant (retrieval mode):\n\n"
            + context
            + "\n\nAdd a Groq API key to .env to enable generated answers."
        )

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a real estate assistant. Answer clearly "
                            "using the provided knowledge. If the information "
                            "is insufficient, say so. Do not invent prices."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Retrieved knowledge:\n{context}\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
                "temperature": 0.3,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except Exception:
        return (
            "AI service is currently unavailable. Retrieved information:\n\n"
            + context
        )

@app.post("/chat", response_class=HTMLResponse)
def chat(
    request: Request,
    question: str = Form(...),
):
    df = get_properties()
    question = question.strip()

    if question:
        answer = answer_question(question)
    else:
        answer = "Please enter a question."

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "total_properties": len(df),
            "average_price": int(df["SalePrice"].mean())
                if not df.empty and "SalePrice" in df.columns else 0,
            "prediction": None,
            "recommendations": [],
            "error": None,
            "chat_question": question,
            "chat_answer": answer,
        },
    )

@app.get("/health")
def health():
    return {
        "status": "running",
        "model_available": os.path.exists(MODEL_PATH),
        "dataset_available": os.path.exists(DATA_PATH),
    }

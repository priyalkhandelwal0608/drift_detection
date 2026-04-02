from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
INDEX_HTML = os.path.join(BASE_DIR, "templates", "index.html")

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
model = joblib.load(MODEL_PATH)

# FastAPI app
app = FastAPI(title="Fraud Detection API")

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Function to read template and optionally insert result
def render_html(result_html: str = "") -> str:
    with open(INDEX_HTML, "r") as f:
        html_content = f.read()
    # Replace placeholder with result (empty if first visit)
    html_content = html_content.replace("{{ result }}", result_html)
    return html_content

# Homepage route
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content=render_html())

# Prediction route
@app.post("/predict", response_class=HTMLResponse)
def predict(
    transaction_amount: float = Form(...),
    account_age_days: int = Form(...),
    num_transactions: int = Form(...)
):
    # Make prediction
    X = [[transaction_amount, account_age_days, num_transactions]]
    prediction = model.predict(X)[0]

    # Prepare result HTML
    if prediction == 1:
        result_html = '<div class="result fraud">Fraudulent Transaction</div>'
    else:
        result_html = '<div class="result legit">Legitimate Transaction</div>'

    # Return HTML with result inserted
    return HTMLResponse(content=render_html(result_html))
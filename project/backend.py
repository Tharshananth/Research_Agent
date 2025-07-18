from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow Streamlit to connect
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def handle_query(request: Request):
    data = await request.json()
    user_query = data.get("query")

    # Replace this with your real AI logic
    summary = f"This is a generated summary for: {user_query}"
    pdf_url = "https://www.example.com/sample_summary.pdf"
    links = [
        "https://arxiv.org/abs/1234.5678",
        "https://scholar.google.com/scholar?q=AI"
    ]

    return JSONResponse({
        "summary": summary,
        "pdf_url": pdf_url,
        "links": links
    })
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware  # ← NEW IMPORT

app = FastAPI(title="Genesis Backend")

# 👇 CORS MIDDLEWARE 👇
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 👆 END CORS 👆

# ... rest of your code (HF_TOKEN, routes, etc.)
# Get Hugging Face token from environment
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN is missing! Set it in Render.")

HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
OSDR_BASE = "https://osdr.nasa.gov/osdr/api/v1"

class SearchRequest(BaseModel):
    query: str

class StudyCard(BaseModel):
    id: str
    title: str
    organism: str
    mission: str
    assay_type: str
    principal_investigator: str
    osdr_url: str

class ApiResponse(BaseModel):
    query: str
    ai_summary: str
    statistics: Dict[str, Any]
    study_cards: List[StudyCard]

async def call_mistral(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 250, "temperature": 0.1}
            }
        )
        if resp.status_code != 200:
            raise HTTPException(500, f"HF error: {resp.text}")
        return resp.json()[0]["generated_text"].strip()

@app.post("/api/search")
async def search(request: SearchRequest):
    # STEP 1: Parse query
    parse_prompt = f'Extract JSON: organism, factor. Query: "{request.query}"'
    try:
        parsed_str = await call_mistral(parse_prompt)
        # Very simple JSON extraction (for demo)
        organism = "Arabidopsis thaliana" if "arabidopsis" in request.query.lower() else "Mus musculus"
        factor = "space radiation" if "radiation" in request.query.lower() else "microgravity"
    except:
        organism, factor = "Arabidopsis thaliana", "space radiation"

    # STEP 2: Mock OSDR response (for testing)
    mock_studies = [
        {
            "id": "OSD-123",
            "title": f"Transcriptomic Response of {organism} to {factor}",
            "organism": organism,
            "mission": "ISS",
            "assay_type": "RNA-Seq",
            "principal_investigator": "Dr. NASA Scientist",
        }
    ]

    study_cards = [
        StudyCard(
            id=s["id"],
            title=s["title"],
            organism=s["organism"],
            mission=s["mission"],
            assay_type=s["assay_type"],
            principal_investigator=s["principal_investigator"],
            osdr_url=f"https://osdr.nasa.gov/study/{s['id']}"
        )
        for s in mock_studies
    ]

    # STEP 3: Mock AI summary
    ai_summary = f"Found 1 study on {organism} under {factor} conditions aboard the ISS."

    return ApiResponse(
        query=request.query,
        ai_summary=ai_summary,
        statistics={
            "total_studies": 1,
            "top_organism": organism,
            "missions": ["ISS"],
            "assays": ["RNA-Seq"]
        },
        study_cards=study_cards
    )

@app.get("/health")
async def health():
    return {"status": "✅ Backend is LIVE!"}
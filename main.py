from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import json
from typing import List, Dict, Any

# ======================
# CONFIGURATION
# ======================
app = FastAPI(title="Genesis: NASA Space Biology Search Engine")

# CORS — allow frontend (localhost + Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",           # Local dev
        "https://genesis-frontend.vercel.app",  # Replace with your Vercel URL
        "*"  # For demo flexibility (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN is missing! Set it in Render environment variables.")

HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
OSDR_BASE_URL = "https://osdr.nasa.gov/osdr/api/v1"

# ======================
# MODELS
# ======================
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

# ======================
# HELPERS
# ======================
async def call_mistral(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.1,
                    "return_full_text": False
                }
            }
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Hugging Face API error: {response.text}"
            )
        result = response.json()
        return result[0]["generated_text"].strip()

def parse_study(osdr_data: Dict) -> StudyCard:
    """Convert raw OSDR study metadata into a clean StudyCard."""
    return StudyCard(
        id=osdr_data.get("id", "N/A"),
        title=osdr_data.get("title", "Untitled Study"),
        organism=osdr_data.get("organism", "Unknown"),
        mission=osdr_data.get("mission", "Unknown"),
        assay_type=osdr_data.get("assay_type", "Unknown"),
        principal_investigator=osdr_data.get("principal_investigator", "Anonymous"),
        osdr_url=f"https://osdr.nasa.gov/osdr/studies/{osdr_data.get('id', '')}"
    )

# ======================
# ROUTES
# ======================
@app.post("/api/search", response_model=ApiResponse)
async def search_nasa_biology(request: SearchRequest):
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # === STEP 1: Parse query with Mistral ===
    parse_prompt = f"""
    Extract biological search parameters from this query.
    Output ONLY valid JSON with these keys (omit if unknown): organism, factor, tissue, mission.
    Use scientific names (e.g., "Homo sapiens", not "human").
    Query: "{user_query}"
    """
    try:
        raw_output = await call_mistral(parse_prompt)
        
        # Clean potential markdown
        if raw_output.startswith("```json"):
            json_str = raw_output.split("```json", 1)[1].split("```", 1)[0]
        elif raw_output.startswith("```"):
            json_str = raw_output.split("```", 1)[1].split("```", 1)[0]
        else:
            json_str = raw_output
        
        parsed = json.loads(json_str)
    except Exception as e:
        # Fallback: use basic keyword matching
        parsed = {}
        if "arabidopsis" in user_query.lower():
            parsed["organism"] = "Arabidopsis thaliana"
        elif "mouse" in user_query.lower() or "mice" in user_query.lower():
            parsed["organism"] = "Mus musculus"
        if "radiation" in user_query.lower():
            parsed["factor"] = "space radiation"
        elif "microgravity" in user_query.lower():
            parsed["factor"] = "microgravity"

    # === STEP 2: Query NASA OSDR ===
    filters = []
    if "organism" in parsed:
        filters.append(f'organism:"{parsed["organism"]}"')
    if "factor" in parsed:
        filters.append(f'factor:"{parsed["factor"]}"')
    if "mission" in parsed:
        filters.append(f'mission:"{parsed["mission"]}"')
    
    if not filters:
        return ApiResponse(
            query=user_query,
            ai_summary="Could not extract meaningful search terms. Try: 'Arabidopsis space radiation' or 'mouse microgravity ISS'.",
            statistics={"total_studies": 0, "top_organism": "N/A", "missions": [], "assays": []},
            study_cards=[]
        )

    query_string = " AND ".join(filters)
    osdr_query_url = f"{OSDR_BASE_URL}/query?q={query_string}&format=json"

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            osdr_resp = await client.get(osdr_query_url)
            osdr_resp.raise_for_status()
            osdr_data = osdr_resp.json()
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"NASA OSDR API error: {str(e)}"
            )

    # Normalize response (OSDR sometimes returns list, sometimes dict)
    studies_list = osdr_data if isinstance(osdr_data, list) else osdr_data.get("studies", [])
    if not studies_list:
        return ApiResponse(
            query=user_query,
            ai_summary="No relevant studies found in NASA's Open Science Data Repository.",
            statistics={"total_studies": 0, "top_organism": "N/A", "missions": [], "assays": []},
            study_cards=[]
        )

    # Fetch full metadata for top 5 studies
    study_ids = [s.get("id") for s in studies_list[:5] if s.get("id")]
    full_studies = []
    for sid in study_ids:
        try:
            study_resp = await client.get(f"{OSDR_BASE_URL}/studies/{sid}")
            if study_resp.status_code == 200:
                full_studies.append(study_resp.json())
        except:
            continue

    study_cards = [parse_study(s) for s in full_studies]

    # === STEP 3: Generate AI summary ===
    if not study_cards:
        ai_summary = "Studies found, but metadata could not be retrieved."
        stats = {"total_studies": 0, "top_organism": "N/A", "missions": [], "assays": []}
    else:
        context = "\n".join([
            f"- '{s.title}' ({s.organism}, Mission: {s.mission}, Assay: {s.assay_type})"
            for s in study_cards
        ])
        summary_prompt = f"""
        Summarize these NASA space biology studies in 2-3 clear sentences.
        Then output ONLY a JSON object with: total_studies (int), top_organism (str), missions (list of str), assays (list of str).
        Studies:
        {context}
        """
        ai_output = await call_mistral(summary_prompt)

        # Extract summary (before JSON)
        if "{" in ai_output:
            ai_summary = ai_output.split("{")[0].strip()
            # TODO: Parse JSON stats properly in v2
        else:
            ai_summary = ai_output

        # Basic stats (improve later with JSON parsing)
        stats = {
            "total_studies": len(study_cards),
            "top_organism": study_cards[0].organism,
            "missions": list(set(s.mission for s in study_cards)),
            "assays": list(set(s.assay_type for s in study_cards))
        }

    return ApiResponse(
        query=user_query,
        ai_summary=ai_summary,
        statistics=stats,
        study_cards=study_cards
    )

@app.get("/health")
async def health_check():
    return {"status": "✅ Genesis Backend is LIVE", "osdr_api": OSDR_BASE_URL}
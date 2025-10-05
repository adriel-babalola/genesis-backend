from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import json
from typing import List, Dict, Any
import urllib.parse

# ======================
# CONFIGURATION
# ======================
app = FastAPI(title="Genesis: NASA Space Biology Search Engine")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN is missing! Set it in Render environment variables.")

HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# NASA OSDR API endpoints (CORRECT ONES)
OSDR_SEARCH_URL = "https://osdr.nasa.gov/osdr/data/search"
OSDR_META_URL = "https://osdr.nasa.gov/osdr/data/osd/meta"

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
    """Call Mistral AI via Hugging Face"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
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
                print(f"Mistral API error: {response.text}")
                return ""
            result = response.json()
            return result[0]["generated_text"].strip()
        except Exception as e:
            print(f"Mistral call failed: {e}")
            return ""

def extract_study_metadata(study_data: Dict) -> StudyCard:
    """Extract study metadata from NASA OSDR response"""
    try:
        # Get study comments which contain key metadata
        study_info = study_data.get("study", {})
        if isinstance(study_info, dict):
            study_key = list(study_info.keys())[0]  # e.g., "OSD-137"
            study_obj = study_info[study_key]
        else:
            study_obj = study_data
        
        # Extract from 'studies' array
        studies = study_obj.get("additionalInformation", {}).get("description", {}).get("studies", [{}])
        if studies:
            main_study = studies[0]
        else:
            main_study = study_obj
        
        # Get study ID
        study_id = study_obj.get("identifier", "N/A")
        
        # Get title
        title = main_study.get("title", "Untitled Study")
        
        # Get organism from comments
        organism = "Unknown"
        mission = "Unknown"
        assay_type = "Unknown"
        pi = "Anonymous"
        
        comments = main_study.get("comments", [])
        for comment in comments:
            name = comment.get("name", "")
            value = comment.get("value", "")
            if name == "Mission Name":
                mission = value
        
        # Get organism from organisms section
        organisms_data = study_obj.get("additionalInformation", {}).get("organisms", {})
        if organisms_data:
            organism_keys = list(organisms_data.get("links", {}).keys())
            if organism_keys:
                organism = organism_keys[0].replace("_", " ").title()
        
        # Get assay type from assays
        assays = study_obj.get("additionalInformation", {}).get("assays", {})
        if assays:
            assay_keys = list(assays.keys())
            if assay_keys:
                # Extract technology type from assay name
                assay_name = assay_keys[0]
                if "rna-seq" in assay_name.lower():
                    assay_type = "RNA-Seq"
                elif "microarray" in assay_name.lower():
                    assay_type = "Microarray"
                elif "mass-spec" in assay_name.lower():
                    assay_type = "Mass Spectrometry"
                elif "microscopy" in assay_name.lower():
                    assay_type = "Microscopy"
                else:
                    assay_type = assay_name.split("_")[1] if "_" in assay_name else "Unknown"
        
        # Get PI from people
        people = main_study.get("people", [])
        for person in people:
            if "Principal Investigator" in person.get("roles", []):
                pi = f"{person.get('firstName', '')} {person.get('lastName', '')}".strip()
                break
        
        return StudyCard(
            id=study_id,
            title=title,
            organism=organism,
            mission=mission,
            assay_type=assay_type,
            principal_investigator=pi,
            osdr_url=f"https://osdr.nasa.gov/bio/repo/data/studies/{study_id}"
        )
    except Exception as e:
        print(f"Error parsing study: {e}")
        return StudyCard(
            id="N/A",
            title="Error parsing study",
            organism="Unknown",
            mission="Unknown",
            assay_type="Unknown",
            principal_investigator="Unknown",
            osdr_url=""
        )

# ======================
# ROUTES
# ======================
@app.post("/api/search", response_model=ApiResponse)
async def search_nasa_biology(request: SearchRequest):
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # === STEP 1: Parse query with Mistral (optional - fallback to keyword matching) ===
    parsed = {}
    query_lower = user_query.lower()
    
    # Simple keyword extraction
    if "arabidopsis" in query_lower:
        parsed["organism"] = "Arabidopsis thaliana"
    elif "mouse" in query_lower or "mice" in query_lower:
        parsed["organism"] = "Mus musculus"
    elif "human" in query_lower:
        parsed["organism"] = "Homo sapiens"
    
    if "radiation" in query_lower:
        parsed["term"] = "radiation"
    elif "microgravity" in query_lower:
        parsed["term"] = "microgravity"
    elif "bone" in query_lower:
        parsed["term"] = "bone"
    elif "muscle" in query_lower:
        parsed["term"] = "muscle"
    
    # === STEP 2: Query NASA OSDR Search API ===
    search_params = {
        "type": "cgene",  # NASA OSDR database
        "size": 10  # Number of results
    }
    
    # Build search query
    if parsed.get("organism"):
        search_params["ffield"] = "organism"
        search_params["fvalue"] = parsed["organism"]
    
    if parsed.get("term"):
        search_params["term"] = parsed["term"]
    
    if not search_params.get("term") and not search_params.get("ffield"):
        # Use entire query as search term
        search_params["term"] = user_query
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Search for studies
            search_response = await client.get(OSDR_SEARCH_URL, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            # Extract study IDs from search results
            hits = search_data.get("hits", {}).get("hits", [])
            if not hits:
                return ApiResponse(
                    query=user_query,
                    ai_summary="No studies found matching your query. Try different keywords like 'mouse', 'radiation', or 'ISS'.",
                    statistics={"total_studies": 0, "top_organism": "N/A", "missions": [], "assays": []},
                    study_cards=[]
                )
            
            # Get detailed metadata for top 5 studies
            study_cards = []
            for hit in hits[:5]:
                source = hit.get("_source", {})
                study_id = source.get("Accession", "").replace("GLDS-", "OSD-")
                
                if not study_id or study_id == "OSD-":
                    continue
                
                try:
                    # Fetch full metadata
                    study_num = study_id.replace("OSD-", "")
                    meta_url = f"{OSDR_META_URL}/{study_num}"
                    meta_response = await client.get(meta_url)
                    
                    if meta_response.status_code == 200:
                        meta_data = meta_response.json()
                        study_card = extract_study_metadata(meta_data)
                        study_cards.append(study_card)
                except Exception as e:
                    print(f"Error fetching metadata for {study_id}: {e}")
                    continue
            
            if not study_cards:
                return ApiResponse(
                    query=user_query,
                    ai_summary="Studies found but metadata could not be retrieved. Try a more specific query.",
                    statistics={"total_studies": 0, "top_organism": "N/A", "missions": [], "assays": []},
                    study_cards=[]
                )
            
            # === STEP 3: Generate AI summary ===
            context = "\n".join([
                f"- {s.title} (Organism: {s.organism}, Mission: {s.mission})"
                for s in study_cards
            ])
            
            summary_prompt = f"""Summarize these NASA space biology studies in 2-3 sentences:
{context}
Focus on the main research themes and organisms studied."""
            
            ai_summary = await call_mistral(summary_prompt)
            if not ai_summary:
                ai_summary = f"Found {len(study_cards)} studies related to {user_query}. These studies investigate biological responses to spaceflight conditions."
            
            # Calculate statistics
            organisms = [s.organism for s in study_cards if s.organism != "Unknown"]
            missions = [s.mission for s in study_cards if s.mission != "Unknown"]
            assays = [s.assay_type for s in study_cards if s.assay_type != "Unknown"]
            
            stats = {
                "total_studies": len(study_cards),
                "top_organism": organisms[0] if organisms else "Unknown",
                "missions": list(set(missions)) if missions else ["Unknown"],
                "assays": list(set(assays)) if assays else ["Unknown"]
            }
            
            return ApiResponse(
                query=user_query,
                ai_summary=ai_summary,
                statistics=stats,
                study_cards=study_cards
            )
            
        except Exception as e:
            print(f"NASA OSDR API error: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"NASA OSDR API error: {str(e)}"
            )

@app.get("/")
async def root():
    return {"message": "Genesis Backend is LIVE", "status": "✅"}

@app.get("/health")
async def health_check():
    return {
        "status": "✅ Genesis Backend is LIVE",
        "osdr_search_api": OSDR_SEARCH_URL,
        "osdr_meta_api": OSDR_META_URL
    }

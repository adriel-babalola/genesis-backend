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
# Use a working model - Mistral-7B-Instruct-v0.2 is publicly available
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
    async with httpx.AsyncClient(timeout=60.0) as client:
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
                error_text = response.text
                print(f"Mistral API error [{response.status_code}]: {error_text}")
                # Model might be loading, return empty string to use fallback
                return ""
            result = response.json()
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and "generated_text" in result[0]:
                    return result[0]["generated_text"].strip()
            return ""
        except Exception as e:
            print(f"Mistral call failed: {e}")
            return ""

def extract_study_metadata(study_data: Dict) -> StudyCard:
    """Extract study metadata from NASA OSDR response"""
    try:
        # Navigate to the actual study object
        study_info = study_data.get("study", {})
        
        # Get the first (and usually only) study key like "OSD-137"
        if isinstance(study_info, dict) and study_info:
            study_key = list(study_info.keys())[0]
            study_obj = study_info[study_key]
        else:
            study_obj = study_data
        
        # Get study ID
        study_id = study_obj.get("identifier", "N/A")
        
        # Navigate to the studies array for detailed metadata
        studies_array = []
        try:
            studies_array = study_obj.get("studies", [])
            if not studies_array:
                # Try alternate path
                studies_array = study_obj.get("additionalInformation", {}).get("description", {}).get("studies", [])
        except:
            pass
        
        main_study = studies_array[0] if studies_array else {}
        
        # Get title
        title = main_study.get("title", study_obj.get("title", "Untitled Study"))
        
        # Initialize defaults
        organism = "Unknown"
        mission = "Unknown"
        assay_type = "Unknown"
        pi = "Anonymous"
        
        # Extract from comments
        comments = main_study.get("comments", [])
        for comment in comments:
            name = comment.get("name", "")
            value = comment.get("value", "")
            
            if name == "Mission Name":
                mission = value
            elif name == "Project Title" and mission == "Unknown":
                mission = value
        
        # Get organism - try multiple paths
        try:
            # Path 1: From organisms links
            organisms_data = study_obj.get("additionalInformation", {}).get("organisms", {})
            organism_links = organisms_data.get("links", {})
            if organism_links:
                organism_key = list(organism_links.keys())[0]
                organism = organism_key.replace("musmusculus", "Mus musculus")
                organism = organism.replace("arabidopsisthaliana", "Arabidopsis thaliana")
                organism = organism.replace("homosapiens", "Homo sapiens")
                organism = organism.replace("_", " ").title()
        except:
            pass
        
        # Get assay type - try multiple paths
        try:
            assays = study_obj.get("additionalInformation", {}).get("assays", {})
            if assays:
                assay_keys = list(assays.keys())
                if assay_keys:
                    assay_name = assay_keys[0].lower()
                    
                    if "rna-seq" in assay_name or "transcription-profiling" in assay_name:
                        assay_type = "RNA-Seq"
                    elif "microarray" in assay_name:
                        assay_type = "Microarray"
                    elif "mass-spec" in assay_name or "protein-expression" in assay_name:
                        assay_type = "Mass Spectrometry"
                    elif "microscopy" in assay_name or "imaging" in assay_name:
                        assay_type = "Microscopy"
                    elif "methylation" in assay_name or "bisulfite" in assay_name:
                        assay_type = "DNA Methylation"
                    else:
                        assay_type = assay_name.split("_")[1].replace("-", " ").title() if "_" in assay_name else "Genomics"
        except:
            pass
        
        # Get PI
        people = main_study.get("people", [])
        for person in people:
            roles = person.get("roles", [])
            if any("Principal Investigator" in role for role in roles):
                first = person.get("firstName", "")
                last = person.get("lastName", "")
                pi = f"{first} {last}".strip()
                if pi:
                    break
        
        return StudyCard(
            id=study_id,
            title=title[:200] if title else "Untitled Study",  # Limit title length
            organism=organism,
            mission=mission,
            assay_type=assay_type,
            principal_investigator=pi,
            osdr_url=f"https://osdr.nasa.gov/bio/repo/data/studies/{study_id}"
        )
    except Exception as e:
        print(f"Error parsing study metadata: {e}")
        import traceback
        traceback.print_exc()
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
            
            print(f"NASA Search returned {len(hits)} hits")
            
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
                
                # Try multiple ways to get study ID
                study_id = source.get("Accession", source.get("Study Identifier", ""))
                
                # Clean up ID format
                if study_id.startswith("GLDS-"):
                    study_id = study_id.replace("GLDS-", "OSD-")
                elif not study_id.startswith("OSD-"):
                    study_id = f"OSD-{study_id}" if study_id else ""
                
                print(f"Processing study: {study_id}")
                
                if not study_id or study_id == "OSD-":
                    # Try to extract from hit itself
                    title = source.get("Study Title", "Unknown Study")
                    organism = source.get("organism", "Unknown")
                    mission = source.get("Project Title", "Unknown")
                    assay = source.get("Study Assay Technology Type", "Unknown")
                    pi_list = source.get("Study Publication Author List", "")
                    pi = pi_list.split(",")[0] if pi_list else "Unknown"
                    
                    study_cards.append(StudyCard(
                        id=source.get("Accession", "N/A"),
                        title=title[:200],
                        organism=organism,
                        mission=mission,
                        assay_type=assay,
                        principal_investigator=pi,
                        osdr_url=f"https://osdr.nasa.gov/bio/repo/search?q={urllib.parse.quote(title)}"
                    ))
                    continue
                
                try:
                    # Fetch full metadata
                    study_num = study_id.replace("OSD-", "")
                    meta_url = f"{OSDR_META_URL}/{study_num}"
                    
                    print(f"Fetching metadata from: {meta_url}")
                    
                    meta_response = await client.get(meta_url, timeout=10.0)
                    
                    if meta_response.status_code == 200:
                        meta_data = meta_response.json()
                        study_card = extract_study_metadata(meta_data)
                        
                        # If metadata parsing failed, use search result data
                        if study_card.title == "Untitled Study" or study_card.title == "Error parsing study":
                            study_card.title = source.get("Study Title", study_card.title)[:200]
                            study_card.organism = source.get("organism", study_card.organism)
                            study_card.mission = source.get("Project Title", study_card.mission)
                        
                        study_cards.append(study_card)
                    else:
                        print(f"Metadata fetch failed with status {meta_response.status_code}")
                        # Fall back to search result data
                        study_cards.append(StudyCard(
                            id=study_id,
                            title=source.get("Study Title", "Unknown Study")[:200],
                            organism=source.get("organism", "Unknown"),
                            mission=source.get("Project Title", "Unknown"),
                            assay_type=source.get("Study Assay Technology Type", "Unknown"),
                            principal_investigator=source.get("Study Publication Author List", "").split(",")[0] if source.get("Study Publication Author List") else "Unknown",
                            osdr_url=f"https://osdr.nasa.gov/bio/repo/data/studies/{study_id}"
                        ))
                        
                except Exception as e:
                    print(f"Error fetching metadata for {study_id}: {e}")
                    # Use search result as fallback
                    study_cards.append(StudyCard(
                        id=study_id,
                        title=source.get("Study Title", "Unknown Study")[:200],
                        organism=source.get("organism", "Unknown"),
                        mission=source.get("Project Title", "Unknown"),
                        assay_type=source.get("Study Assay Technology Type", "Unknown"),
                        principal_investigator="Unknown",
                        osdr_url=f"https://osdr.nasa.gov/bio/repo/data/studies/{study_id}"
                    ))
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

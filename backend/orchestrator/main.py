from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie, Document
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uvicorn
import os
from dotenv import load_dotenv
import urllib.parse

# Charger les variables d'environnement
load_dotenv()

# --- MODÈLES MONGODB ---
class Tender(Document):
    title: str
    description: str
    amount: float
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "tenders"

# --- MODÈLES PYDANTIC POUR L'API ---
class TenderCreate(BaseModel):
    title: str
    description: str
    amount: float

# --- GESTIONNAIRE MONGODB ---
class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    
db = MongoDB()

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Récupérer l'URL directement sans modification
        mongodb_url = os.getenv("MONGODB_URL")
        
        print(f"🔄 Tentative de connexion à MongoDB...")
        db.client = AsyncIOMotorClient(mongodb_url)
        
        # Test simple de connexion
        await db.client.admin.command('ping')
        print("✅ MongoDB Atlas connected successfully")
        
        # Initialiser Beanie
        await init_beanie(
            database=db.client.tenderdb,
            document_models=[Tender]
        )
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    if db.client:
        db.client.close()
        
# --- APPLICATION ---
app = FastAPI(
    title="Tender Analysis API",
    version="0.1.0",
    description="Backend for Tender Analysis System",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {
        "message": "Tender Analysis API is running!",
        "status": "healthy",
        "mongodb": "connected" if db.client else "disconnected"
    }

@app.get("/health")
async def health_check():
    # Vérifier MongoDB
    mongo_status = "disconnected"
    try:
        if db.client:
            await db.client.admin.command('ping')
            mongo_status = "connected"
    except:
        pass
    
    return {
        "status": "ok",
        "service": "tender-orchestrator",
        "mongodb": mongo_status
    }

# --- ENDPOINTS TENDER ---
@app.post("/api/tenders", response_model=dict)
async def create_tender(tender_data: TenderCreate):
    """Créer un nouveau tender"""
    try:
        tender = Tender(**tender_data.dict())
        await tender.insert()
        return {
            "id": str(tender.id),
            "status": "created",
            "message": "Tender created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tenders")
async def get_all_tenders():
    """Récupérer tous les tenders"""
    try:
        tenders = await Tender.find_all().to_list()
        return [
            {
                "id": str(tender.id),
                "title": tender.title,
                "description": tender.description,
                "amount": tender.amount,
                "status": tender.status,
                "created_at": tender.created_at.isoformat()
            }
            for tender in tenders
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tenders/{tender_id}")
async def get_tender(tender_id: str):
    """Récupérer un tender par ID"""
    try:
        tender = await Tender.get(tender_id)
        if not tender:
            raise HTTPException(status_code=404, detail="Tender not found")
        
        return {
            "id": str(tender.id),
            "title": tender.title,
            "description": tender.description,
            "amount": tender.amount,
            "status": tender.status,
            "created_at": tender.created_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/tenders/{tender_id}")
async def delete_tender(tender_id: str):
    """Supprimer un tender"""
    try:
        tender = await Tender.get(tender_id)
        if not tender:
            raise HTTPException(status_code=404, detail="Tender not found")
        
        await tender.delete()
        return {"status": "deleted", "message": "Tender deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT TEST MONGODB ---
@app.get("/api/test-mongo")
async def test_mongodb():
    """Tester la connexion MongoDB"""
    try:
        if not db.client:
            return {"status": "error", "message": "MongoDB not connected"}
        
        # Test ping
        await db.client.admin.command('ping')
        
        # Compter les documents
        count = await Tender.count()
        
        return {
            "status": "success",
            "message": "MongoDB is working",
            "tenders_count": count
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
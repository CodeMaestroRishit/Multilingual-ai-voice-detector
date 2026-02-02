from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List
from app.api.auth import verify_api_key
from app.utils.audio import process_audio_input
from app.models.detector import get_detector

router = APIRouter()

class DetectRequest(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    transcript: Optional[str] = None 
    message: Optional[str] = None

# Hackathon Exact Format
class DetectResponse(BaseModel):
    classification: str       # "AI" | "Human"
    confidence: float         # 0.0 - 1.0
    explanation: str
    fraud_risk: str           # "HIGH" | "LOW"
    risk_keywords: List[str]
    overall_risk: str         # "CRITICAL" | "SAFE"
    transcript_preview: str   # Added Feature

@router.post("/detect", response_model=DetectResponse, dependencies=[Depends(verify_api_key)])
async def detect_voice(request: DetectRequest):
    if not request.audio_base64 and not request.audio_url:
        raise HTTPException(
            status_code=400, 
            detail="Must provide either 'audio_base64' or 'audio_url'"
        )
    
    try:
        audio_array = process_audio_input(request.audio_base64, request.audio_url)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

    try:
        detector = get_detector()
        result = detector.detect_fraud(audio_array, provided_transcript=request.transcript)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

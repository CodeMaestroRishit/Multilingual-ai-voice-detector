import torch
import numpy as np
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, pipeline
from torch.nn import CosineSimilarity

class VoiceDetector:
    _instance = None
    
    def __init__(self):
        # 1. Primary AI Voice Detector (Acoustic Features)
        self.wav2vec_name = "facebook/wav2vec2-large-xlsr-53"
        print(f"Loading Acoustic Model: {self.wav2vec_name} ...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.wav2vec_name)
        self.model = Wav2Vec2Model.from_pretrained(self.wav2vec_name)
        self.model.eval()
        
        # 2. Secondary Semantics & Cadence Model (Whisper)
        print("Loading Semantic Model (Whisper Small)...")
        # UPGRADE: Using "small" for better embeddings/multilingual support
        self.transcriber = pipeline(
            "automatic-speech-recognition", 
            model="openai/whisper-small"
        )
        print("All Models loaded successfully.")
        
        self.fraud_keywords = {
            "financial": ["bank", "account", "credit", "debit", "card", "cvv", "pin", "verify", "blocked", "suspended"],
            "urgency": ["immediately", "urgent", "expires", "arrest", "police", "legal", "warrant", "action"],
            "scams": ["lottery", "winner", "refund", "gift card", "tech support", "virus", "hacked", "lucky", "draw"]
        }

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _calculate_smoothness(self, embeddings: torch.Tensor) -> float:
        """
        Calculates temporal smoothness.
        AI voices tend to have higher frame-to-frame cosine similarity (less 'jitter').
        """
        # Embeddings shape: [1, Sequence_Length, Hidden_Size]
        # We compare frame T with frame T+1
        if embeddings.shape[1] < 2:
            return 0.0
            
        cos = CosineSimilarity(dim=1, eps=1e-6)
        # Compare all frames with their next frame
        similarity = cos(embeddings[0, :-1, :], embeddings[0, 1:, :])
        return float(similarity.mean().item())

    def detect_fraud(self, audio_array: np.ndarray, provided_transcript: str = None):
        """
        Hackathon Logic:
        1. Primary: AI Voice Classification (from Audio Signal)
        2. Secondary: Fraud Risk (from Keywords)
        """
        
        # --- STEP 1: TRANSCRIPTION (Semantic Layer) ---
        final_transcript = provided_transcript
        if not final_transcript:
            try:
                # Force translate to English so keywords work universally
                # Input: simple numpy array (float32)
                result = self.transcriber(audio_array.astype(np.float32), generate_kwargs={"task": "translate"})
                final_transcript = result.get("text", "")
            except Exception as e:
                print(f"Transcription Error: {e}")
                final_transcript = ""

        # --- STEP 2: ACOUSTIC ANALYSIS (AI Detection) ---
        inputs = self.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state # Shape: [1, Time, 1024]
        
        # Metric A: Variance (Richness)
        # Real voices have high variance (ups/downs). AI is often "flatter" or more uniform.
        np_embeds = embeddings[0].numpy()
        time_variance = np.var(np_embeds, axis=0).mean()
        
        # Metric B: Smoothness (Robotic-ness)
        # AI voices often transition too smoothly between phonemes.
        smoothness = self._calculate_smoothness(embeddings)
        
        # --- SCORING LOGIC ---
        # Heuristic: High Smoothness + Low Variance -> AI
        # Normalizing scores (approximate ranges based on XLS-R-53 behavior)
        # Variance usually 0.005 (low) to 0.05 (high)
        # Smoothness usually 0.90 (rough) to 0.99 (smooth)
        
        conf_smooth = (smoothness - 0.90) * 10 # Scale up small differences
        conf_var = 1.0 - (time_variance * 20)  # Invert variance
        
        ai_confidence = (conf_smooth + conf_var) / 2.0
        ai_confidence = float(np.clip(ai_confidence, 0.0, 1.0))

        classification = "AI" if ai_confidence > 0.5 else "Human"
        
        explanation = []
        if conf_smooth > 0.6:
            explanation.append("High temporal smoothness (robotic consistency)")
        if conf_var > 0.6:
            explanation.append("Low embedding variance (lack of natural emotion)")
        if not explanation:
            explanation.append("Natural voice patterns detected")

        # --- STEP 3: CONTENT RISK (Keywords) ---
        keyword_hits = []
        if final_transcript:
            text = final_transcript.lower()
            for category, words in self.fraud_keywords.items():
                for word in words:
                    if word in text:
                        keyword_hits.append(word) # Just the word

        fraud_risk = "LOW"
        if len(keyword_hits) >= 1:
            fraud_risk = "MEDIUM"
        if len(keyword_hits) >= 3 or ("otp" in keyword_hits) or ("bank" in keyword_hits):
            fraud_risk = "HIGH"

        # --- STEP 4: OVERALL RISK ---
        overall_risk = "SAFE"
        if classification == "AI" and fraud_risk != "LOW":
             overall_risk = "CRITICAL" # Al + Scam Words = Dangerous
        elif classification == "AI":
             overall_risk = "WARNING" # AI but innocuous text
        elif fraud_risk == "HIGH":
             overall_risk = "WARNING" # Human but asking for OTP

        return {
            "classification": classification,
            "confidence": round(ai_confidence, 2),
            "explanation": " + ".join(explanation),
            "fraud_risk": fraud_risk,
            "risk_keywords": list(set(keyword_hits)), # Unique
            "overall_risk": overall_risk,
            "transcript_preview": final_transcript # Added as requested
        }

# Global instance
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = VoiceDetector.get_instance()
    return detector

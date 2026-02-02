# SecureCall - AI Voice Fraud Detector

A Cybersecurity API that detects:
1.  **AI/Deepfake Voices** (using Wav2Vec2 audio analysis).
2.  **Scam/Fraud Keywords** (using OpenAI Whisper for transcription).
3.  **Urgency/Aggression** (using audio signal processing).

## 🚀 Setup & Run

### Prerequisites
1.  **Python 3.10+**
2.  **FFmpeg** (Required for audio processing)

---

### Mac / Linux Setup
1.  **Install FFmpeg**:
    ```bash
    brew install ffmpeg
    # OR
    sudo apt install ffmpeg
    ```

2.  **Setup Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Run Server**:
    ```bash
    uvicorn app.main:app --reload
    ```

---

### Windows Setup (For Team)
1.  **Install FFmpeg**:
    *   **Option A (Winget)**: Run `winget install "FFmpeg (Essentials)"` in PowerShell.
    *   **Option B (Chocolatey)**: Run `choco install ffmpeg`.
    *   **Option C (Manual)**: Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract, and add the `bin` folder to your System PATH.

2.  **Setup Environment**:
    Open PowerShell or Command Prompt:
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
    *Note: If you get a permission error on step 2, run `Set-ExecutionPolicy Unrestricted -Scope Process` first.*

3.  **Run Server**:
    ```powershell
    uvicorn app.main:app --reload
    ```

---

## 🧪 How to Test

### 1. Verification Script (Easiest)
We have a Python script that automatically handles the API request.

**Run on Mac/Linux:**
```bash
python3 test_api.py
```

**Run on Windows:**
```powershell
python test_api.py
```

*To test your own local file, open `test_api.py` and edit the file path at the bottom.*

### 2. Manual Test (Base64 / URL)
**Endpoint**: `POST /detect`
**Header**: `X-API-Key: demo_key_123`

```json
{
  "audio_url": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Example.ogg",
  "transcript": "Optional text override"
}
```

## 🛡️ Response format
```json
{
  "threat_level": "High",
  "is_fraud": true,
  "alert": "CRITICAL: High-risk call detected...",
  "transcript_preview": "TEXT HEARD BY MODEL...",
  "analysis": {
    "voice_type": "AI",
    "sentiment": "Urgent/Aggressive",
    "keywords_detected": ["urgency:arrest"]
  }
}
```

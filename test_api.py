import requests
import sys
import base64

API_URL = "http://localhost:8000"
API_KEY = "demo_key_123"

def print_result(data, file_label):
    print(f"\n📊 ANALYSIS REPORT ({file_label})")
    print("--------------------------------------------------")
    print(f"🎙️  PRIMARY CHECK (Voice Line)")
    print(f"    Classification : {data['classification'].upper()}")
    print(f"    Confidence     : {int(data['confidence'] * 100)}%")
    print(f"    Reasoning      : {data['explanation']}")
    print("--------------------------------------------------")
    print(f"🛡️  SECONDARY CHECK (Content Line)")
    print(f"    Fraud Risk     : {data['fraud_risk']}")
    print(f"    Keywords Found : {data['risk_keywords']}")
    print("--------------------------------------------------")
    print(f"🚨 OVERALL VERDICT : {data['overall_risk']}")
    print(f"📝 TRANSCRIPT PREVIEW:")
    print(f"   \"{data['transcript_preview']}\"")
    print("--------------------------------------------------\n")

def test_detect_url():
    print("\nTesting /detect (URL Mode)...")
    payload = {
         # Benign Wikipedia Audio
        "audio_url": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Example.ogg",
    }
    headers = {"X-API-Key": API_KEY, "User-Agent": "TestScript"}
    
    try:
        r = requests.post(f"{API_URL}/detect", json=payload, headers=headers)
        if r.status_code == 200:
            print_result(r.json(), "URL: Wikipedia OGG")
        else:
            print(f"❌ Failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_detect_local_file(file_path):
    print(f"\nTesting /detect (Local File: {file_path})...")
    
    try:
        with open(file_path, "rb") as f:
            audio_content = f.read()
            encoded_string = base64.b64encode(audio_content).decode('utf-8')
            
        payload = {"audio_base64": encoded_string}
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return

    headers = {"X-API-Key": API_KEY, "User-Agent": "TestScript"}
    
    try:
        r = requests.post(f"{API_URL}/detect", json=payload, headers=headers)
        if r.status_code == 200:
            print_result(r.json(), f"File: {file_path.split('/')[-1]}")
        else:
            print(f"❌ Failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Test User's Local File
    test_detect_local_file("/Users/rishitguha/Downloads/ElevenLabs_2026-02-01T13_17_32_Vanishree - Energetic Indian English_pvc_sp100_s50_sb75_se0_b_m2.mp3")
    
    # Optional: Test URL
    # test_detect_url()

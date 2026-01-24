import json
import os
import time
import asyncio
import httpx
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
# Using the OpenAI-compatible endpoint for Google AI Studio
URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MODELS_FILE = "google_free_models.json"
TEST_PROMPT = "Explain the concept of 'Vibes' in programming in 2 sentences."

async def test_model(client, model_id, semaphore):
    # Clean model ID: remove 'models/' prefix if present for the API call
    clean_id = model_id.replace("models/", "")
    
    async with semaphore:
        result = {
            "model_id": model_id,
            "status": "failed",
            "latency": 0,
            "ttft": 0,
            "tps": 0,
            "tokens": 0,
            "error": None
        }
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": clean_id,
            "messages": [{"role": "user", "content": TEST_PROMPT}],
            "stream": True 
        }
        
        start_time = time.time()
        ttft = 0
        tokens_count = 0
        full_content = ""
        
        try:
            async with client.stream("POST", URL, headers=headers, json=payload, timeout=30.0) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    result["error"] = f"HTTP {response.status_code}: {error_text.decode()}"
                    return result
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if ttft == 0:
                                ttft = time.time() - start_time
                            
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta and delta['content']:
                                    full_content += delta['content']
                                    tokens_count += 1
                        except:
                            continue
            
            end_time = time.time()
            duration = end_time - start_time
            
            result["status"] = "success"
            result["latency"] = round(duration, 3)
            result["ttft"] = round(ttft, 3)
            result["tokens"] = tokens_count
            # TPS calculation: tokens / (total_time - time_to_first_token)
            processing_time = duration - ttft
            result["tps"] = round(tokens_count / processing_time, 2) if processing_time > 0 else 0
            
        except Exception as e:
            result["error"] = str(e)
            
        print(f"[{result['status'].upper()}] {model_id} - {result['latency']}s, {result['tps']} tok/s")
        return result

async def main():
    if not API_KEY:
        print("Error: GOOGLE_API_KEY not found in .env")
        return

    try:
        with open(MODELS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {MODELS_FILE}: {e}")
        return
    
    models = [m["model"] for m in data.get("data", [])]
    print(f"Found {len(models)} Google models to test.")
    
    # Google AI Studio has generous free tier but let's be respectful
    semaphore = asyncio.Semaphore(2) 
    
    results = []
    async with httpx.AsyncClient() as client:
        tasks = [test_model(client, model_id, semaphore) for model_id in models]
        results = await asyncio.gather(*tasks)
    
    # Final Report
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provider": "Google AI Studio",
        "total_models": len(models),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "results": results
    }
    
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"google_test_results_{ts_str}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {filename}")

if __name__ == "__main__":
    asyncio.run(main())

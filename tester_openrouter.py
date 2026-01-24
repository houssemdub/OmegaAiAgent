import json
import os
import time
import asyncio
import httpx
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_FILE = "openrouter_free_models.json"
TEST_PROMPT = "Explain the concept of 'Vibes' in programming in 2 sentences."

async def test_model(client, model_id, semaphore):
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
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/houssemdub/OmegaAiAgent",
            "X-Title": "OmegaAi Model Tester"
        }
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": TEST_PROMPT}],
            "stream": True # We use streaming to measure TTFT and TPS
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
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if ttft == 0:
                                ttft = time.time() - start_time
                            
                            delta = data['choices'][0]['delta']
                            if 'content' in delta and delta['content']:
                                full_content += delta['content']
                                tokens_count += 1 # Rough estimate if usage isn't in stream
                            
                            # Check for usage in stream (some providers send it at the end)
                            if 'usage' in data:
                                tokens_count = data['usage']['total_tokens']
                        except:
                            continue
            
            end_time = time.time()
            duration = end_time - start_time
            
            result["status"] = "success"
            result["latency"] = round(duration, 3)
            result["ttft"] = round(ttft, 3)
            result["tokens"] = tokens_count
            result["tps"] = round(tokens_count / (duration - ttft), 2) if (duration - ttft) > 0 else 0
            
        except Exception as e:
            result["error"] = str(e)
            
        print(f"[{result['status'].upper()}] {model_id} - {result['latency']}s, {result['tps']} tok/s")
        return result

async def main():
    if not API_KEY:
        print("Error: OPENROUTER_API_KEY not found in .env")
        return

    with open(MODELS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    models = [m["id"] for m in data.get("data", [])]
    print(f"Found {len(models)} models to test.")
    
    # We use a semaphore to avoid hitting OpenRouter's rate limits too aggressively
    semaphore = asyncio.Semaphore(3) 
    
    results = []
    async with httpx.AsyncClient() as client:
        tasks = [test_model(client, model_id, semaphore) for model_id in models]
        results = await asyncio.gather(*tasks)
    
    # Final Report
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_models": len(models),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "results": results
    }
    
    filename = f"openrouter_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {filename}")

if __name__ == "__main__":
    asyncio.run(main())

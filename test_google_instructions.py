import os
import json
import asyncio
from pathlib import Path
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

async def test_model_instructions(model_id, api_key):
    client = genai.Client(api_key=api_key)
    try:
        # Try with system instruction
        config = genai_types.GenerateContentConfig(
            system_instruction="You are a helpful assistant.",
            temperature=0.1
        )
        response = client.models.generate_content(
            model=model_id.replace("models/", ""),
            contents="Say 'Instruction OK'",
            config=config
        )
        return {"status": "PASS", "message": response.text.strip()}
    except Exception as e:
        err_msg = str(e)
        if "instruction" in err_msg.lower():
            return {"status": "FAIL_INSTRUCTION", "error": err_msg}
        return {"status": "FAIL_OTHER", "error": err_msg}

async def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found in environment.")
        return

    json_path = Path("google_free_models.json")
    if not json_path.exists():
        print("google_free_models.json not found.")
        return

    with open(json_path, "r") as f:
        models_data = json.load(f).get("data", [])

    results = []
    print(f"Testing {len(models_data)} models for Instruction Compatibility...")

    for m in models_data:
        m_id = m["model"]
        print(f"Testing {m_id}...", end=" ", flush=True)
        res = await test_model_instructions(m_id, api_key)
        print(res["status"])
        
        results.append({
            "model": m_id,
            "display_name": m["display_name"],
            "instruction_support": res["status"],
            "details": res.get("message") or res.get("error")
        })
        # Avoid rate limits
        await asyncio.sleep(1)

    output_path = Path("google_instruction_test_results.json")
    with open(output_path, "w") as f:
        json.dump({"test_date": "2026-01-22", "results": results}, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())

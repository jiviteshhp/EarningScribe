import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag import get_or_create_collection, retrieve

BASE = os.path.expanduser("~/EarningScribe")

# We use the small model locally just to test the pipeline works.
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"


def load_model(model_name=MODEL_NAME):
    print(f"Loading model: {model_name}")

    # The tokenizer converts text to numbers 
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # The model itself — loaded in float32 for CPU
    # On GPU - float16 or bfloat16 to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cpu"
    )

    print("Model loaded.")
    return tokenizer, model


def build_prompt(transcript, retrieved_contexts):
    # Quality of the prompt directly determines quality of output.
    
    # We follow the structure from the LAOS paper:
    # 1. Role — tell the model who it is
    # 2. Context — inject RAG retrieved documents
    # 3. Task — tell it exactly what to produce
    # 4. Input — the actual transcript
    # 5. Format — tell it the exact output structure

    # Format retrieved contexts as a numbered list
    context_text = ""
    for i, ctx in enumerate(retrieved_contexts):
        context_text += f"\n[Context {i+1} - {ctx['ticker']} {ctx['date']}]\n"
        context_text += ctx["document"][:300]  # limit each context chunk

    prompt = f"""You are a senior financial analyst specializing in earnings call analysis.

RELEVANT CONTEXT FROM SIMILAR EARNINGS CALLS:
{context_text}

TASK: Analyze the following earnings call transcript and generate a structured report.
Return your response as valid JSON with exactly these fields:
- "company_summary": 2-3 sentence overview of company performance
- "key_metrics": list of specific numbers mentioned (revenue, margins, growth rates)
- "guidance": what management said about future quarters
- "risks": main risks or challenges mentioned
- "sentiment": one of "positive", "neutral", or "negative"

TRANSCRIPT:
{transcript[:1500]}

STRUCTURED REPORT (JSON only, no other text):"""

    return prompt


def generate_report(tokenizer, model, prompt):
    # Tokenize the prompt — convert text to token IDs
    inputs = tokenizer(
        prompt,
        return_tensors="pt",  # pt = PyTorch tensors
        truncation=True,
        max_length=2048
    )

    # Generate output tokens
    # max_new_tokens: how many tokens to generate (not including input)
    # temperature: randomness. 0.1 = very focused/deterministic output
    # do_sample: whether to sample randomly or take the top prediction
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode only the newly generated tokens (not the input prompt)
    # outputs[0] is the first (and only) sequence
    # inputs["input_ids"].shape[1] is the length of the input
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return response


def parse_json_response(response):
    # The model should return JSON but sometimes adds extra text.
    # This function extracts just the JSON part robustly.
    try:
        # Try direct parse first
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # If that fails, find the JSON block between { and }
    start = response.find("{")
    end   = response.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

    # If all parsing fails, return raw text so we don't crash
    return {"raw_response": response}


def run_pipeline(transcript, query=None):

    # retrieve relevant context
    collection = get_or_create_collection()
    query_text = query if query else transcript[:200]
    retrieved  = retrieve(collection, query_text, top_k=3)

    # build prompt with context injected
    prompt = build_prompt(transcript, retrieved)

    # load model and generate
    tokenizer, model = load_model()
    raw_response     = generate_report(tokenizer, model, prompt)

    # parse and return structured output
    report = parse_json_response(raw_response)

    return {
        "report":            report,
        "retrieved_context": [
            {"ticker": r["ticker"], "date": r["date"], "score": r["score"]}
            for r in retrieved
        ],
        "raw_response": raw_response
    }


if __name__ == "__main__":
    # Load a real sample from test set to run through the pipeline
    test_path = os.path.join(BASE, "data", "processed", "test.json")
    with open(test_path) as f:
        test_data = json.load(f)

    # Take the first test sample
    sample = test_data[0]

    print("=" * 60)
    print(f"Running pipeline on sample: {sample['id']}")
    print(f"Ticker: {sample.get('ticker', 'unknown')}")
    print("=" * 60)

    result = run_pipeline(sample["transcript"])

    print("\n--- RETRIEVED CONTEXT USED ---")
    for ctx in result["retrieved_context"]:
        print(f"  {ctx['ticker']} | {ctx['date']} | similarity: {ctx['score']}")

    print("\n--- GENERATED REPORT ---")
    print(json.dumps(result["report"], indent=2))

    print("\n--- RAW MODEL OUTPUT ---")
    print(result["raw_response"])

    # Save result for inspection
    output_path = os.path.join(BASE, "results", "sample_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResult saved to {output_path}")
    print("done.")
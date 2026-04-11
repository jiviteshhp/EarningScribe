import json
import os
from datasets import load_dataset

BASE = os.path.expanduser("~/EarningScribe")

def download_and_save():
    print("Downloading lamini/earnings-calls-qa...")

    
    dataset = load_dataset("lamini/earnings-calls-qa")

    print("\nDataset structure:")
    print(dataset)

    
    sample = dataset["train"][0]
    print("\n--- SAMPLE KEYS ---")
    print(list(sample.keys()))

    print("\n--- TRANSCRIPT (first 500 chars) ---")
    print(sample["transcript"][:500])

    print("\n--- QUESTION ---")
    print(sample["question"])

    print("\n--- ANSWER ---")
    print(sample["answer"])

    return dataset


def analyse_lengths(dataset):
    print("\n--- LENGTH ANALYSIS ---")

    for split in dataset.keys():
        transcripts = dataset[split]["transcript"]
        t_lengths = [len(t.split()) for t in transcripts]

        print(f"\n{split.upper()} ({len(transcripts)} samples)")
        print(f"  Transcript words — avg: {sum(t_lengths)//len(t_lengths)}, "
              f"max: {max(t_lengths)}, "
              f"min: {min(t_lengths)}")


def clean_text(text):
    
    lines = text.split("\n")
    cleaned = []

    skip_phrases = [
        "operator",
        "safe harbor",
        "forward-looking",
        "this concludes",
        "thank you for joining",
        "ladies and gentlemen",
        "this call is being recorded",
    ]

    for line in lines:
        line = line.strip()
        if len(line) < 15:
            continue
        if any(line.lower().startswith(p) for p in skip_phrases):
            continue
        cleaned.append(line)

    return " ".join(cleaned)


def build_structured_output(question, answer):
    return {
        "financial_insight": question,
        "detail": answer
    }

def process_and_save(dataset):
    output_dir = os.path.join(BASE, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    all_samples = dataset["train"]
    clean_samples = []
    # Track seen transcripts to avoid duplicates
    seen_transcripts = set()

    for i, item in enumerate(all_samples):
        answer = item["answer"].strip()
        transcript = clean_text(item["transcript"])

        if "i do not know" in answer.lower():
            continue
        if len(transcript.split()) < 50:
            continue
        if len(answer.split()) < 5:
            continue

        # Use first 200 chars as a fingerprint
        # Two transcripts with same opening are the same document
        fingerprint = transcript[:200]
        if fingerprint in seen_transcripts:
            continue
        seen_transcripts.add(fingerprint)

        clean_samples.append({
            "id":         f"sample_{i}",
            "transcript": transcript,
            "ticker":     item.get("ticker", ""),
            "date":       item.get("date", ""),
            "output": {
                "financial_insight": item["question"].strip(),
                "detail":            answer
            }
        })

        if len(clean_samples) == 5000:
            break

    total = len(clean_samples)
    train_end = int(total * 0.8)
    val_end   = int(total * 0.9)

    splits = {
        "train":      clean_samples[:train_end],
        "validation": clean_samples[train_end:val_end],
        "test":        clean_samples[val_end:]
    }

    for split_name, samples in splits.items():
        save_path = os.path.join(output_dir, f"{split_name}.json")
        with open(save_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Saved {len(samples)} samples -> {save_path}")


def verify_saved_data():
    print("\n--- VERIFYING SAVED DATA ---")

    for fname in os.listdir(os.path.join(BASE, "data", "processed")):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(BASE, "data", "processed", fname)
        with open(path) as f:
            data = json.load(f)

        sample = data[0]
        print(f"\n{fname}: {len(data)} samples")
        print(f"  Transcript preview : {sample['transcript'][:120]}...")
        print(f"  Structured output  : {json.dumps(sample['output'], indent=4)}")


if __name__ == "__main__":
    dataset = download_and_save()
    analyse_lengths(dataset)
    process_and_save(dataset)
    verify_saved_data()
    print("\n Data is ready.")
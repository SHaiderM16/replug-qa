import string
import hashlib
import json
import time
from pathlib import Path
from collections import Counter


def normalize(text: str) -> Counter:
    # Lowercase, strip punctuation from each token, split on whitespace
    # No stopword removal (conserve meaning)
    # Returns Counter to preserve token multiplicities for F1
    tokens = text.lower().split()
    cleaned_tokens = []
    for token in tokens:
        cleaned = token.strip(string.punctuation)
        if cleaned:
            cleaned_tokens.append(cleaned)
    return Counter(cleaned_tokens)


def normalize_str(text: str) -> str:
    # Lowercase, strip punctuation, remove articles (a, an, the)
    # Returns normalized string for Exact Match evaluation
    tokens = text.lower().split()
    cleaned_tokens = []
    articles = {'a', 'an', 'the'}
    for token in tokens:
        cleaned = token.strip(string.punctuation)
        if cleaned and cleaned not in articles:
            cleaned_tokens.append(cleaned)
    return ' '.join(cleaned_tokens)


def make_cache_key(prompt: str, passage: str = "") -> str:
    # Create unique cache key from prompt and optional passage
    content = f"{prompt}|{passage}"
    return hashlib.md5(content.encode()).hexdigest()


def load_cache(cache_file):
    # Load JSONL cache file into dict
    cache = {}
    if Path(cache_file).exists():
        with open(cache_file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                key = entry["cache_key"]
                cache[key] = entry
    return cache


def save_cache_entry(cache_file, entry):
    # Append entry to cache file
    with open(cache_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def groq_api_call_with_retry(client, prompt, max_retries=5):
    # Manual retry loop for Groq API calls
    delays = [1, 2, 4, 8, 16]

    for attempt, delay in enumerate(delays[:max_retries]):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=64,
            )
            return {
                "answer": response.choices[0].message.content,
            }
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            print(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
            time.sleep(delay)

    return None

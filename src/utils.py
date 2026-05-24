import string
import hashlib
import json
import time
import os
from pathlib import Path
from collections import Counter
import cohere


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
    # Standard NQ normalization: lowercase, strip punctuation, remove articles
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


# Cache format version - increment when structure changes
CACHE_VERSION = 2


def load_cache(cache_file):
    # Load JSONL cache file into dict
    cache = {}
    if Path(cache_file).exists():
        with open(cache_file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                # Skip entries from old cache versions (missing total_logprob)
                if entry.get("cache_version") != CACHE_VERSION:
                    continue
                key = entry["cache_key"]
                cache[key] = entry
    return cache


def save_cache_entry(cache_file, entry):
    # Append entry to cache file with version tag
    entry["cache_version"] = CACHE_VERSION
    with open(cache_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def cohere_api_call_with_retry(client, prompt, max_retries=5):
    # Manual retry loop for Cohere API calls
    delays = [1, 2, 4, 8, 16]
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    # Rate limit throttle: Cohere free tier = 20 req/min → one call every 3 seconds
    time.sleep(3)

    for attempt, delay in enumerate(delays[:max_retries]):
        try:
            response = co.chat(
                model="command-r-plus-08-2024",
                messages=[{"role": "user", "content": prompt}],
                logprobs=True,
                temperature=0,
                max_tokens=15,
            )

            # Extract answer and total logprob
            # Response structure varies: sometimes 1 item (thinking), sometimes 2 items (thinking + text)
            answer = None

            # First try to find a direct text content item
            for content_item in response.message.content:
                if hasattr(content_item, 'text') and hasattr(content_item, 'type') and content_item.type == 'text':
                    answer = content_item.text
                    break

            # If no direct text item, extract from thinking content
            if answer is None:
                for content_item in response.message.content:
                    if hasattr(content_item, 'thinking'):
                        thinking_text = str(content_item.thinking)
                        # Try to extract answer from thinking content
                        # Look for patterns like "answer is X" or direct statements
                        import re
                        # Try to find the last sentence, which is usually the answer
                        sentences = thinking_text.split('.')
                        if sentences:
                            # Get the last complete sentence
                            potential_answer = sentences[-1].strip()
                            if potential_answer and len(potential_answer) > 3:
                                answer = potential_answer + "."
                        # Fallback: use the full thinking content
                        if not answer:
                            answer = thinking_text[:200]  # Truncate if too long
                        break

            # Ultimate fallback
            if not answer:
                answer = str(response.message.content[0])

            # Remove trailing punctuation for cleaner EM matching
            answer = answer.rstrip('.,!?;:')

            # Compute total logprob if logprobs exist
            if hasattr(response, 'logprobs') and response.logprobs:
                total_logprob = sum(token.logprobs[0] for token in response.logprobs)
            else:
                total_logprob = float("-inf")

            return {
                "answer": answer,
                "total_logprob": total_logprob
            }
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            print(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
            time.sleep(delay)

    return None

#!/usr/bin/env python3
"""
Quick test for OpenAI-style advanced params: logprobs, top_logprobs, logit_bias, frequency_penalty.

Usage:
  # Start Parallax HTTP server first (e.g. on port 8000), then:
  python -m tests.test_openai_advanced_params
  # Or with custom base URL:
  PARALLAX_BASE_URL=http://127.0.0.1:8000 python -m tests.test_openai_advanced_params

Requires: requests
"""

import json
import os
import sys

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

# Default Parallax chat completions endpoint
BASE_URL = os.environ.get("PARALLAX_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
URL = f"{BASE_URL}/v1/chat/completions"


def test_stream_advanced_params():
    """Stream request with logprobs=True, top_logprobs=3, logit_bias={13: -100}, frequency_penalty=1.2."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "stream": True,
        "max_tokens": 32,
        "logprobs": True,
        "top_logprobs": 3,
        "logit_bias": {"13": -100},
        "frequency_penalty": 1.2,
    }

    print(f"POST {URL}")
    print("Request body:", json.dumps(payload, indent=2))
    print("-" * 60)

    try:
        r = requests.post(
            URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=30,
        )
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        if hasattr(e, "response") and e.response is not None:
            print("Response body:", e.response.text[:500], file=sys.stderr)
        sys.exit(1)

    full_content = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or line.strip() != line:
            continue
        if line.startswith("data: "):
            data = line[6:].strip()
            if data == "[DONE]":
                print("\n[DONE]")
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                print(f"Skip non-JSON line: {data[:80]}...")
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                full_content.append(content)
                print(content, end="", flush=True)
            logprobs = choices[0].get("logprobs")
            if logprobs and "content" in logprobs and logprobs["content"]:
                entry = logprobs["content"][0]
                print(
                    f"\n  [logprob={entry.get('logprob')} token={repr(entry.get('token'))} "
                    f"top_logprobs={len(entry.get('top_logprobs', []))} items]",
                    end="",
                )
            if choices[0].get("finish_reason"):
                print(f"\nfinish_reason: {choices[0]['finish_reason']}")

    print("-" * 60)
    print("Full reply:", "".join(full_content))
    print("OK: stream with advanced params completed.")


if __name__ == "__main__":
    test_stream_advanced_params()


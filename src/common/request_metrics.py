import json


def get_request_metrics(chunk, start_time, first_token_time, last_token_time):
    try:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        if isinstance(chunk, str):
            chunk = chunk.removeprefix("data: ").lstrip()
            chunk = json.loads(chunk)
        usage = chunk.get("usage")
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        usage.get("total_tokens")
        tps = output_tokens / (last_token_time - first_token_time)
        ttft = int((first_token_time - start_time) * 1000)
        return tps, ttft, input_tokens, output_tokens
    except Exception:
        return None, None, None, None

#!/usr/bin/env python3
import requests
import json
import sys

def main():
    user_input = sys.argv[1] if len(sys.argv) > 1 else "Hi"

    url = "http://localhost:2000/v1/chat/completions"

    payload = {
        "messages": [{"role": "user", "content": user_input}],
        "stream": True,                # 启用流式
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": True},
        "sampling_params": {"top_k": 1}
    }

    headers = {"Content-Type": "application/json"}

    try:
        # 启用流式请求
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            response.raise_for_status()

            # 逐行读取响应（SSE 格式：data: {...}\n\n）
            for line in response.iter_lines():
                if line:
                    # SSE 数据以 "data: " 开头
                    if line.startswith(b'data: '):
                        json_str = line[len(b'data: '):].decode('utf-8')
                        if json_str.strip() == '[DONE]':
                            break

                        try:
                            chunk = json.loads(json_str)
                            # 提取 content（兼容 OpenAI 格式）
                            delta = chunk['choices'][0]['delta']
                            content = delta.get('content', '')
                            if content:
                                print(content, end='', flush=True)
                        except (KeyError, json.JSONDecodeError) as e:
                            # 可选：打印错误或跳过
                            continue
        print()  # 最后换行

    except requests.exceptions.RequestException as e:
        print(f"\n请求失败: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
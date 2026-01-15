#!/bin/bash

# 设置 API 地址
URL="http://localhost:2000/v1/chat/completions"

# 构造 JSON 数据（注意：必须正确转义引号）
DATA='{
    "messages": [
        {
            "role": "user",
            "content": "北京天气"
        }
    ],
    "stream": false,
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": true},
    "sampling_params": {
        "top_k": 1
    }
}'

# 发送请求
curl --location \
     --header 'Content-Type: application/json' \
     --data "$DATA" \
     "$URL"




# curl --location 'http://localhost:3210/v1/chat/completions' \
#   --header 'Content-Type: application/json' \
#   --data '{
#     "max_tokens": 10,
#     "messages": [
#       {
#         "role": "user",
#         "content": [
#           {
#             "type": "image_url",
#             "image_url": {
#               "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
#             }
#           },
#           {"type": "text", "text": "Describe this image."}
#         ]
#       }
#     ],
#     "stream": false
#   }'


curl --location 'http://localhost:3210/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data '{
    "max_tokens": 10,
    "messages": [
      {
        "role": "user",
        "content": "hello"
      }
    ],
    "stream": false
  }'

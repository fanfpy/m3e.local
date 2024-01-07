## 基于m3e 兼容openapi的text-embedding-ada-002接口服务

## 使用方法

```shell
git clone https://github.com/fanfpy/m3e.local.git
cd m3e.local
docker-compose up -d
# running on http://0.0.0.0:6006
```

## 验证

### request
```shell
curl --location 'http://127.0.0.1:6006/v1/embeddings' \
--header 'Authorization: Bearer sk-aaabbbcccdddeeefffggghhhiiijjjkkk' \
--header 'Content-Type: application/json' \
--data '{
    "input": ["hello m3e"],
    "model": "text-embedding-ada-002",
    "encoding_format": "float"
  }'

#response


```
### response
```json
{
    "data": [
        {
            "embedding": [
                0.04027857258915901,
                0.005487577989697456,
                -0.025278501212596893,
                -0.004541480913758278 ...
            ],
            "index": 0,
            "object": "embedding"
        }
    ],
    "model": "text-embedding-ada-002",
    "object": "list",
    "usage": {
        "prompt_tokens": 2,
        "total_tokens": 4
    }
}
```
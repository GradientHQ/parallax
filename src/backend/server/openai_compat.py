import base64
import json
from typing import Any, Dict, Optional, Tuple

from fastapi.responses import JSONResponse

PARALLAX_HTTP_RESPONSE_ENVELOPE = "__parallax_http_response__"


def openai_error_payload(
    message: str,
    *,
    err_type: str = "server_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code or err_type,
        }
    }


def openai_error_response(
    message: str,
    *,
    status_code: int,
    err_type: str = "server_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> JSONResponse:
    return JSONResponse(
        content=openai_error_payload(
            message,
            err_type=err_type,
            param=param,
            code=code,
        ),
        status_code=status_code,
    )


def openai_models_payload(model_name: Optional[str]) -> Dict[str, Any]:
    models = []
    if model_name:
        models.append(
            {
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "parallax",
            }
        )
    return {"object": "list", "data": models}


def encode_http_response_envelope(
    *,
    status_code: int,
    content_type: Optional[str],
    body: bytes,
) -> bytes:
    envelope = {
        PARALLAX_HTTP_RESPONSE_ENVELOPE: True,
        "status_code": int(status_code),
        "content_type": content_type or "application/json",
        "body_base64": base64.b64encode(body).decode("ascii"),
    }
    return json.dumps(envelope, separators=(",", ":")).encode("utf-8")


def decode_http_response_envelope(content: bytes) -> Optional[Tuple[int, str, bytes]]:
    try:
        envelope = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(envelope, dict) or envelope.get(PARALLAX_HTTP_RESPONSE_ENVELOPE) is not True:
        return None

    body_base64 = envelope.get("body_base64")
    if not isinstance(body_base64, str):
        return None

    try:
        body = base64.b64decode(body_base64)
    except ValueError:
        return None

    status_code = int(envelope.get("status_code", 200))
    content_type = envelope.get("content_type") or "application/json"
    return status_code, str(content_type), body

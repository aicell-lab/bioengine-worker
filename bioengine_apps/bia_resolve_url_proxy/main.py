import base64
import os
from typing import Any
from urllib.parse import urlparse

import httpx
from hypha_rpc.utils.schema import schema_method
from ray import serve

_DEFAULT_ALLOWED_HOSTS = "beta.bioimagearchive.org,www.ebi.ac.uk,ftp.ebi.ac.uk"


def _allowed_hosts() -> set[str]:
    raw_value = os.environ.get("RESOLVE_URL_ALLOWED_HOSTS", _DEFAULT_ALLOWED_HOSTS)
    return {part.strip().lower() for part in raw_value.split(",") if part.strip()}


def _normalize_headers(headers: dict[str, Any] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    if not isinstance(headers, dict):
        return normalized
    for key, value in headers.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str):
            normalized[key] = value
        elif value is not None:
            normalized[key] = str(value)
    return normalized


@serve.deployment(
    ray_actor_options={
        "num_cpus": 0.5,
        "num_gpus": 0,
        "runtime_env": {
            "pip": [
                "httpx",
            ],
        },
    }
)
class ResolveUrlProxy:
    @schema_method
    async def resolve_url(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, Any] | None = None,
        timeout: float = 60.0,
        body: str | dict[str, Any] | list[Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            parsed = urlparse(str(url))
        except Exception as exp:
            return {
                "ok": False,
                "status_code": 400,
                "url": str(url),
                "error": f"Invalid URL: {exp}",
            }

        if parsed.scheme.lower() != "https":
            return {
                "ok": False,
                "status_code": 400,
                "url": str(url),
                "error": "Only https URLs are allowed",
            }

        host = (parsed.hostname or "").lower()
        if host not in _allowed_hosts():
            return {
                "ok": False,
                "status_code": 403,
                "url": str(url),
                "error": f"Host '{host}' is not allowed",
            }

        method_value = str(method or "GET").upper()
        if method_value not in {
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "HEAD",
            "OPTIONS",
        }:
            return {
                "ok": False,
                "status_code": 400,
                "url": str(url),
                "error": f"Unsupported method: {method_value}",
            }

        request_kwargs: dict[str, Any] = {
            "method": method_value,
            "url": str(url),
            "headers": _normalize_headers(headers),
        }
        request_kwargs["headers"].setdefault(
            "User-Agent", "bioengine-bia-resolve-url-proxy/1.0"
        )

        if body is not None:
            if isinstance(body, (dict, list)):
                request_kwargs["json"] = body
            else:
                request_kwargs["content"] = str(body)

        try:
            async with httpx.AsyncClient(
                timeout=max(1.0, float(timeout)),
                follow_redirects=True,
            ) as client:
                response = await client.request(**request_kwargs)
        except Exception as exp:
            return {
                "ok": False,
                "status_code": 502,
                "url": str(url),
                "error": str(exp),
            }

        status_code = int(response.status_code)
        ok = 200 <= status_code < 300
        return {
            "ok": ok,
            "status_code": status_code,
            "url": str(response.url),
            "headers": dict(response.headers),
            "content_base64": base64.b64encode(response.content).decode("ascii"),
            "error": "" if ok else f"Upstream returned HTTP {status_code}",
        }

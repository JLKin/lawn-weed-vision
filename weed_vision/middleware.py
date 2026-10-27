"""Request IDs and a total-body limit enforced before multipart parsing."""

from uuid import uuid4

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .errors import error_body


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, max_request_bytes: int):
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", request_id.encode("ascii"))
                ]
            await send(message)

        async def reject(status: int, code: str, message: str) -> None:
            response = JSONResponse(error_body(request_id, code, message), status_code=status)
            await response(scope, receive, send_with_id)

        if scope["method"] == "POST" and scope["path"] == "/v1/weed-segmentation":
            headers = dict(scope.get("headers", []))
            content_length = headers.get(b"content-length")
            if content_length is not None:
                try:
                    length = int(content_length)
                    if length < 0:
                        raise ValueError
                except ValueError:
                    await reject(400, "invalid_content_length", "Invalid Content-Length header.")
                    return
                if length > self.max_request_bytes:
                    await reject(413, "request_too_large", "The request exceeds the configured byte limit.")
                    return

            # A bounded buffer also protects requests with no Content-Length or a false length.
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > self.max_request_bytes:
                    await reject(413, "request_too_large", "The request exceeds the configured byte limit.")
                    return
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
            replayed = False

            async def replay() -> Message:
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay, send_with_id)
        else:
            await self.app(scope, receive, send_with_id)

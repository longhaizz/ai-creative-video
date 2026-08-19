"""Stop an oversized upload before it reaches the disk.

The size check inside save_upload() is too late to protect anything.
Starlette reads and spools the whole body before the endpoint runs, and
before the token is checked, so a 4 GB request writes 4 GB to the temp
folder and only then gets its 413. Measured, not guessed: an unauthenticated
60 MB upload sent all 60 MB and the temp folder grew by all of it.

That means anyone who can reach the port can fill the disk without a token.
This middleware runs before any of it.

A reverse proxy with client_max_body_size does the same job and is the
usual place for it. This lives here so the server is safe on its own, with
uvicorn exposed directly.
"""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.responses import JSONResponse


class BodySizeLimit:
    """Reject a request body larger than `limit` bytes.

    Two checks, because one is not enough:

    * Content-Length, when the client sends it. This rejects at once, before
      a single byte of body is read.
    * A running count while the body streams. A chunked upload has no
      Content-Length, and a client is free to send a header that lies.
    """

    def __init__(self, app, limit: int):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        declared = headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.limit:
                    await self._refuse(scope, send)
                    return
            except ValueError:
                pass  # a broken header is the body counter's problem

        read = 0
        too_big = False

        async def counted_receive():
            nonlocal read, too_big
            message = await receive()
            if message["type"] == "http.request":
                read += len(message.get("body", b""))
                if read > self.limit:
                    too_big = True
                    # Tell the app the body ended here. It will fail to parse
                    # the form and answer 4xx, and nothing more is written.
                    return {"type": "http.disconnect"}
            return message

        await self.app(scope, counted_receive, send)

    async def _refuse(self, scope, send):
        megabytes = self.limit // (1024 * 1024)
        response = JSONResponse(
            {"detail": f"The request is larger than the {megabytes} MB limit"},
            status_code=413,
        )
        await response(scope, self._empty_receive, send)

    @staticmethod
    async def _empty_receive():
        return {"type": "http.request", "body": b"", "more_body": False}

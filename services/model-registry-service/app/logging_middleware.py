import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

SERVICE_NAME = "model-registry"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        request.state.request_id = request_id
        
        start_time = time.time()
        
        print(f"[{SERVICE_NAME}] [{request_id}] --> {request.method} {request.url.path}")
        
        response = await call_next(request)
        
        duration_ms = (time.time() - start_time) * 1000
        
        print(f"[{SERVICE_NAME}] [{request_id}] <-- {request.method} {request.url.path} | Status: {response.status_code} | Duration: {duration_ms:.2f}ms")
        
        response.headers["X-Request-ID"] = request_id
        
        return response


def log_info(request_id: str, message: str):
    print(f"[{SERVICE_NAME}] [{request_id}] {message}")


def log_error(request_id: str, message: str):
    print(f"[{SERVICE_NAME}] [{request_id}] ERROR: {message}")

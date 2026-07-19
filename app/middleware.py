import uuid
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger()

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            process_time = time.perf_counter() - start_time
            
            # Bind response details
            structlog.contextvars.bind_contextvars(
                status_code=response.status_code,
                duration_ms=round(process_time * 1000, 2)
            )
            
            logger.info("request_completed")
            
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as e:
            process_time = time.perf_counter() - start_time
            structlog.contextvars.bind_contextvars(
                status_code=500,
                duration_ms=round(process_time * 1000, 2),
                error=str(e)
            )
            logger.error("request_failed", exc_info=True)
            raise

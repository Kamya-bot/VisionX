"""
VisionX - End-to-End Request Tracking & Logging
Production-grade observability for ML APIs
"""

import uuid
import time
import logging
from datetime import datetime
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import json

logger = logging.getLogger(__name__)


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """
    Tracks every API request with unique ID for end-to-end observability
    
    Features:
    - Generates unique request_id for each API call
    - Logs request details (method, path, body)
    - Logs response status and duration
    - Tracks ML model used and prediction results
    - Enables distributed tracing
    
    Interview Talking Point:
    "I implemented request tracing middleware to enable end-to-end observability,
    making it easy to debug issues and track ML predictions through the entire pipeline."
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process each request with tracking"""
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Record start time
        start_time = time.time()
        
        # Extract request info
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)
        client_host = request.client.host if request.client else "unknown"
        
        # Log incoming request
        logger.info(
            f"📥 REQUEST START | ID: {request_id[:8]} | "
            f"{method} {path} | Client: {client_host}"
        )
        
        # Log query params if present
        if query_params:
            logger.debug(f"   Query: {query_params}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response
            log_level = logging.INFO if response.status_code < 400 else logging.WARNING
            logger.log(
                log_level,
                f"📤 REQUEST END | ID: {request_id[:8]} | "
                f"Status: {response.status_code} | "
                f"Duration: {duration_ms:.2f}ms"
            )
            
            # Add request ID to response headers (for client-side tracking)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            
            return response
            
        except Exception as e:
            # Log errors with request ID
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"❌ REQUEST ERROR | ID: {request_id[:8]} | "
                f"{method} {path} | Error: {str(e)} | "
                f"Duration: {duration_ms:.2f}ms"
            )
            raise


class MLPredictionLogger:
    """
    Specialized logger for ML predictions
    
    Logs:
    - Input features
    - Model version used
    - Prediction output
    - Confidence score
    - Processing time
    - User context
    """
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.logger = logging.getLogger(f"ml_predictions.{request_id[:8]}")
    
    def log_prediction(
        self,
        user_id: str,
        model_type: str,
        input_features: dict,
        prediction: any,
        confidence: float,
        duration_ms: float,
        model_version: str = "v1.0"
    ):
        """
        Log ML prediction with full context
        
        This creates a complete audit trail for ML decisions
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': self.request_id,
            'user_id': user_id,
            'model_type': model_type,
            'model_version': model_version,
            'input_features': input_features,
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'processing_time_ms': round(duration_ms, 2)
        }
        
        self.logger.info(f"ML_PREDICTION | {json.dumps(log_data)}")
    
    def log_drift_check(
        self,
        drift_detected: bool,
        feature_drift_count: int,
        prediction_psi: float,
        severity: str
    ):
        """Log drift detection results"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': self.request_id,
            'event_type': 'drift_check',
            'drift_detected': drift_detected,
            'feature_drift_count': feature_drift_count,
            'prediction_psi': round(prediction_psi, 4),
            'severity': severity
        }
        
        self.logger.warning(f"DRIFT_CHECK | {json.dumps(log_data)}")
    
    def log_model_load(
        self,
        model_type: str,
        model_version: str,
        load_time_ms: float,
        success: bool
    ):
        """Log model loading events"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': self.request_id,
            'event_type': 'model_load',
            'model_type': model_type,
            'model_version': model_version,
            'load_time_ms': round(load_time_ms, 2),
            'success': success
        }
        
        level = logging.INFO if success else logging.ERROR
        self.logger.log(level, f"MODEL_LOAD | {json.dumps(log_data)}")


def get_request_id(request: Request) -> str:
    """Get request ID from request state"""
    return getattr(request.state, 'request_id', 'unknown')


def get_ml_logger(request: Request) -> MLPredictionLogger:
    """Get ML prediction logger for current request"""
    request_id = get_request_id(request)
    return MLPredictionLogger(request_id)


# Structured logging formatter
class StructuredLogFormatter(logging.Formatter):
    """
    Custom formatter that outputs JSON-structured logs
    
    Benefits:
    - Easy to parse by log aggregators (ELK, Splunk)
    - Searchable by fields
    - Machine-readable
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_production_logging(log_dir: str = "backend/logs"):
    """
    Configure production-grade logging
    
    Sets up:
    - Structured JSON logs
    - Separate files for different log levels
    - Request tracking
    - ML prediction audit trail
    """
    import os
    from logging.handlers import RotatingFileHandler
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler (human-readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler - All logs (JSON structured)
    all_logs_handler = RotatingFileHandler(
        os.path.join(log_dir, 'visionx_all.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    all_logs_handler.setLevel(logging.DEBUG)
    all_logs_handler.setFormatter(StructuredLogFormatter())
    root_logger.addHandler(all_logs_handler)
    
    # File handler - Errors only
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, 'visionx_errors.log'),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredLogFormatter())
    root_logger.addHandler(error_handler)
    
    # File handler - ML predictions only
    ml_handler = RotatingFileHandler(
        os.path.join(log_dir, 'ml_predictions.log'),
        maxBytes=20*1024*1024,  # 20MB
        backupCount=10
    )
    ml_handler.setLevel(logging.INFO)
    ml_handler.setFormatter(StructuredLogFormatter())
    
    # Add to ML prediction logger
    ml_logger = logging.getLogger('ml_predictions')
    ml_logger.addHandler(ml_handler)
    ml_logger.setLevel(logging.INFO)
    
    logger.info("✅ Production logging configured")
    logger.info(f"   Log directory: {log_dir}")
    logger.info(f"   Log files: visionx_all.log, visionx_errors.log, ml_predictions.log")

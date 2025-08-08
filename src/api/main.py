"""
Minimal FastAPI main for production deployment demonstration.
"""

import os
import time
from typing import Dict, Any

# Mock FastAPI for demonstration (in production would import real FastAPI)
class MockFastAPI:
    def __init__(self, title: str = "API", version: str = "0.1.0"):
        self.title = title
        self.version = version
        
    def get(self, path: str):
        def decorator(func):
            return func
        return decorator

# Create app instance
app = MockFastAPI(
    title="Acousto-Gen API",
    version="0.1.0"
)

@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Acousto-Gen API", "version": "0.1.0"}

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "0.1.0"
    }

@app.get("/ready")
def readiness():
    """Readiness check endpoint."""
    return {"status": "ready"}

if __name__ == "__main__":
    print("Acousto-Gen API starting...")
    print("This is a demo implementation - use real FastAPI for production")
"""
RKLLM Server Application Module

This module initializes the Flask application and sets up all routes and middleware.
"""

from flask import Flask
from flask_cors import CORS
import signal
import atexit

from .config import SERVER_HOST, SERVER_PORT, DEBUG_MODE
from .api.routes import setup_routes
from .lib.rkllm_runtime import initialize, cleanup, runtime

def create_app():
    """
    Create and configure the Flask application
    
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    # Set up API routes
    app = setup_routes(app)
    
    # Register shutdown function
    atexit.register(cleanup)
    
    # Handle SIGTERM gracefully
    def handle_sigterm(*args):
        cleanup()
        exit(0)
        
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    return app

def run_server():
    """
    Initialize the RKLLM runtime and start the Flask server
    """
    print("Initializing RKLLM model...")
    if not initialize():
        print("Failed to initialize RKLLM runtime. Exiting.")
        return False
        
    app = create_app()
    
    print(f"Starting Flask server for RKLLM OpenAI-compliant API on http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=DEBUG_MODE, threaded=True)
    
    return True

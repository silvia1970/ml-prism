#!/usr/bin/env python3
"""
PRISM API Start Script
Quick start script for running the PRISM API server
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import and run the app
from api.app import app

if __name__ == '__main__':
    print("=" * 60)
    print("PRISM API Server")
    print("=" * 60)
    print(f"Server starting on http://localhost:5000")
    print(f"API Documentation: http://localhost:5000/apidocs")
    print(f"Health Check: http://localhost:5000/health")
    print("=" * 60)
    print()
    
    # Create necessary directories
    os.makedirs('api_data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run the Flask app (use_reloader=False to prevent infinite restart)
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False
    )

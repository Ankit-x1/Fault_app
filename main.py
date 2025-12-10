import argparse
import sys
from pathlib import Path
import uvicorn
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train import train
from detect import detect

def main():
    parser = argparse.ArgumentParser(
        description='A command-line tool for training an autoencoder model and detecting anomalies in sensor data.'
    )
    parser.add_argument('action', choices=['train', 'detect', 'run_web_app'],
                        help="The action to perform: 'train', 'detect', or 'run_web_app'.")
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help="Host for the web application (default: 0.0.0.0).")
    parser.add_argument('--port', type=int, default=8000,
                        help="Port for the web application (default: 8000).")

    args = parser.parse_args()

    if args.action == 'train':
        print("--- Starting Model Training ---")
        train()
        print("--- Model Training Completed ---")
    elif args.action == 'detect':
        print("--- Starting Fault Detection ---")
        detect()
        print("--- Fault Detection Completed ---")
    elif args.action == 'run_web_app':
        print(f"--- Starting FastAPI Web Application on {args.host}:{args.port} ---")
        # Ensure that fastapi_app.py is in the same directory or accessible via sys.path
        # Importing directly from the file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from fastapi_app import app
        uvicorn.run(app, host=args.host, port=args.port)

if __name__ == '__main__':
    main()

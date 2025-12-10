import argparse
import sys
from pathlib import Path

# Add the 'src' directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train import train
from detect import detect

def main():
    parser = argparse.ArgumentParser(
        description='A command-line tool for training an autoencoder model and detecting anomalies in sensor data.'
    )
    parser.add_argument('action', choices=['train', 'detect'],
                        help="The action to perform: 'train' or 'detect'.")

    args = parser.parse_args()

    if args.action == 'train':
        print("--- Starting Model Training ---")
        train()
        print("--- Model Training Completed ---")
    elif args.action == 'detect':
        print("--- Starting Fault Detection ---")
        detect()
        print("--- Fault Detection Completed ---")

if __name__ == '__main__':
    main()

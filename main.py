import argparse

from scripts import visualize_data, train
import configparser

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run different scripts based on the provided arguments"
    )
    subparsers = parser.add_subparsers(dest="script_name", required=True,
                                       help="Script to run")

    subparsers.add_parser("visualize_data", help="Visualize data")
    subparsers.add_parser("train", help="Train the model")

    args = parser.parse_args()
    if args.script_name == "visualize_data":
        visualize_data()
    if args.script_name == "train":
        train()
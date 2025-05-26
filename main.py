import argparse

from scripts import visualize_data
import configparser

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run different scripts based on the provided arguments"
    )
    subparsers = parser.add_subparsers(dest="script_name", required=True,
                                       help="Script to run")

    parser_visualize_data = subparsers.add_parser("visualize_data", help="Visualize data")

    args = parser.parse_args()
    if args.script_name == "visualize_data":
        visualize_data()
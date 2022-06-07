#!/usr/bin/env python
"""This script """

import argparse
import numpy


def parse_arguments():
    """CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="Time Series Predictive Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count", type=int, required=True, help="Some input from user")
    return parser.parse_args()


def main():
    """Using a main function to avoid global variables"""
    arguments = parse_arguments()
    print(numpy.arange(arguments.count))


if __name__ == "__main__":
    main()

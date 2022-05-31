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
    return parser.parse_args()


def main():
    """Using a main function to avoid global variables"""
    args = parse_arguments()
    print(numpy.arrange(args.count))


if __name__ == "__main__":
    main()

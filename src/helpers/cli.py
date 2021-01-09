import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c', '--config', type=str, help='Path to the config file')
    parser.add_argument('-s', '--scenario', type=str, help='Name of the scenario to run')
    parser.add_argument('-l', '--load', type=str, help='Load a previously trained model')
    parser.add_argument('-f',
                        '--features_only',
                        type=bool,
                        help='Whether to load only the feature extraction part of a previously trained model')

    return parser

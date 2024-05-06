import argparse
from datetime import datetime
import os

from preprocessing.dedup.dedup import exec_deduplication
from preprocessing.filters.pipeline import execute_filtering, execute_url_dedup


def execute_preprocessing(input_dir: str, output_base: str, filtering: bool, dedup: bool):
    os.makedirs(output_base, exist_ok=True)

    if filtering:
        input_dir = execute_url_dedup(input_dir=input_dir, output_base=output_base)
        input_dir = execute_filtering(input_dir=input_dir, output_base=output_base)

    if dedup:
        exec_deduplication(input_dir=input_dir, output_base=output_base)


def arg_parser():
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--input_dir', type=str,
                        help='The input directory containing documents to process', required=True)
    parser.add_argument('--output_dir', type=str,
                        help='The input file containing documents to process', required=False, default="./output")
    parser.add_argument('--filtering', type=bool, help='Whether to execute filtering', required=False, default=True)
    parser.add_argument('--dedup', type=bool, help='Whether to execute deduplication', required=False, default=True)
    return parser.parse_args()


def main():
    args = arg_parser()
    start = datetime.now()
    output_base = os.path.join(args.output_dir, start.strftime("%Y%m%d%H%M%S"))

    execute_preprocessing(args.input_dir, output_base, filtering=args.filtering, dedup=args.dedup)


if __name__ == "__main__":
    main()

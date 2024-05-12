import argparse
from datetime import datetime
import os

from preprocessing.lib import Logger
from preprocessing.dedup.dedup import exec_deduplication
from preprocessing.filters.pipeline import execute_filtering, execute_url_dedup


def execute_preprocessing(input_dir: str, output_base: str, filtering: bool, dedup: bool, *, logger=None):
    logger = logger or Logger.get_logger(__name__, logdir=os.path.join(output_base, "log"))
    if filtering:
        logger.info("Executing url dedup")
        start = datetime.now()
        input_dir = execute_url_dedup(input_dir=input_dir, output_base=output_base, logger=logger)
        end = datetime.now()
        logger.info(f"Finished url dedup in {end - start}")

        logger.info("Executing filtering")
        start = datetime.now()
        input_dir = execute_filtering(input_dir=input_dir, output_base=output_base, logger=logger)
        end = datetime.now()
        logger.info(f"Finished filtering in {end - start}")

    if dedup:
        logger.info("Executing dedup")
        start = datetime.now()
        exec_deduplication(input_dir=input_dir, output_base=output_base, logger=logger)
        end = datetime.now()
        logger.info(f"Finished dedup in {end - start}")


def arg_parser():
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--input_dir', type=str,
                        help='The input directory containing documents to process', required=True)
    parser.add_argument('--output_dir', type=str,
                        help='The input file containing documents to process', required=False, default="./output")
    parser.add_argument('--filtering', type=bool, help='Whether to execute filtering', required=False, default=True)
    parser.add_argument('--dedup', type=bool, help='Whether to execute deduplication', required=False, default=True)
    parser.add_argument('--verbose', type=bool, help='Verbose mode', required=False, default=False)

    return parser.parse_args()


def main():
    args = arg_parser()
    start = datetime.now()
    output_base = os.path.join(args.output_dir, start.strftime("%Y%m%d%H%M%S"))
    logdir = os.path.join(output_base, "log")
    os.makedirs(logdir, exist_ok=True)

    logger = Logger.get_logger(name=__name__, logdir=logdir, verbose=args.verbose)

    execute_preprocessing(args.input_dir, output_base, filtering=args.filtering, dedup=args.dedup, logger=logger)


if __name__ == "__main__":
    main()

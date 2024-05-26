import argparse
import hojichar
from hojichar import document_filters, Compose, Document

import os
import json
import multiprocessing
from datetime import datetime


from preprocessing import lib
from preprocessing.lib import Logger
from preprocessing.filters.document_filters import DiscardMedicalHistory, DiscardCriminalHistory


NUM_WORKER = (multiprocessing.cpu_count() - 2)


def process_filtering(input_file: str, output_base: str, output_file: str, debug: bool = False, run_medical_history: bool = True, run_criminal_history: bool = True, *, logger=None):
    logger = logger or Logger.get_logger(__name__, logdir=os.path.join(output_base, "log"))

    filters = [document_filters.JSONLoader(), document_filters.DocumentNormalizer()]
    if run_medical_history:
        filters.append(DiscardMedicalHistory())
    if run_criminal_history:
        filters.append(DiscardCriminalHistory())
    filters.append(document_filters.JSONDumper(dump_reason=True))
    cleaner = Compose(filters)

    input_doc_iter = (Document(line) for line in lib.readlines(input_file))
    num_jobs = os.environ.get("NUM_WORKER", NUM_WORKER)

    reject_dir = os.path.join(os.path.dirname(output_file), "rejected")
    os.makedirs(reject_dir, exist_ok=True)
    reject_file = os.path.join(reject_dir, os.path.basename(output_file))
    with hojichar.Parallel(cleaner, num_jobs=num_jobs) as filter:
        out_doc_iter = filter.imap_apply(input_doc_iter)

        with open(output_file, "w") as writer:
            with open(reject_file, "w") as reject_writer:
                for result in out_doc_iter:
                    try:
                        if result.is_rejected:
                            reject_writer.write(result.text + "\n")
                        else:
                            writer.write(result.text + "\n")
                    except Exception as e:
                        logger.error(f"Error processing document: {e}")

    if debug:
        os.makedirs(os.path.join(output_base, "stat", "filtering"), exist_ok=True)
        input_file_prefix = os.path.splitext(os.path.basename(input_file))[0]
        with open(os.path.join(output_base, f"{input_file_prefix}.jsonl"), "w") as writer:
            writer.write(json.dumps(cleaner.statistics, ensure_ascii=False) + "\n")


def arg_parser():
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--input_dir', type=str,
                        help='The input directory containing documents to process', required=True)
    parser.add_argument('--output_dir', type=str,
                        help='The input file containing documents to process', required=False, default="./output")
    parser.add_argument('--run_medical_history', type=bool,
                        help='Whether to execute medical history filtering', required=False, default=True)
    parser.add_argument('--run_criminal_history', type=bool,
                        help='Whether to execute criminal history filtering', required=False, default=True)
    parser.add_argument('--debug', type=bool, help='Debug mode', required=False, default=False)
    parser.add_argument('--verbose', type=bool, help='Verbose mode', required=False, default=False)

    return parser.parse_args()


def main():
    args = arg_parser()
    start = datetime.now()
    output_base = os.path.join(args.output_dir, start.strftime("%Y%m%d%H%M%S"))
    logdir = os.path.join(output_base, "log")
    os.makedirs(logdir, exist_ok=True)

    logger = Logger.get_logger(name=__name__, logdir=logdir, verbose=args.verbose)
    for input_file in os.listdir(args.input_dir):
        if not input_file.endswith(".jsonl"):
            continue

        input_full_path = os.path.join(args.input_dir, input_file)

        process_filtering(input_file=input_full_path, output_base=output_base, output_file=os.path.join(output_base, input_file),
                          debug=args.debug, run_medical_history=args.run_medical_history, run_criminal_history=args.run_criminal_history, logger=logger)


if __name__ == "__main__":
    main()

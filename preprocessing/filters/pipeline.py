import hojichar
from hojichar import document_filters, Compose, Document

import os
import json
import multiprocessing


from preprocessing.lib import Logger
from preprocessing.filters.token_filters import RemoveIncompleteSentence, RemoveHeadTailWhitespaceTokenizer, DiscardSpecialCharactersJa, RemoveOnewordNumber
from preprocessing.filters.document_filters import DiscardAdultContentJa, DiscardBBSComments, DiscardDiscriminationContentJa, RemoveRepetition, NewLineSentenceTokenizer, MergeTokens
from preprocessing.dedup.dedup import url_dedup
import preprocessing.lib as lib


NUM_WORKER = (multiprocessing.cpu_count() - 2)


def __output_dir_after_url_dedup(base: str):
    return os.path.join(base, "url_dedup")


def __output_dir_after_filtering(base: str):
    return os.path.join(base, "filtering")


def execute_url_dedup(input_dir: str, output_base: str, *, logger=None) -> str:
    logger = logger or Logger.get_logger(__name__, logdir=os.path.join(output_base, "log"))

    output_dir = __output_dir_after_url_dedup(output_base)
    os.makedirs(output_dir, exist_ok=True)

    for input_file in os.listdir(input_dir):
        if not input_file.endswith(".jsonl"):
            continue

        input_full_path = os.path.join(input_dir, input_file)
        url_dedup(input_file=input_full_path, output_base=output_base,
                  output_file=os.path.join(output_dir, input_file),
                  logger=logger)

    return output_dir


def execute_filtering(input_dir: str, output_base: str, *, logger=None) -> list[str]:
    logger = logger or Logger.get_logger(__name__, logdir=os.path.join(output_base, "log"))

    output_dir = __output_dir_after_filtering(output_base)
    os.makedirs(output_dir, exist_ok=True)

    for input_file in os.listdir(input_dir):
        if not input_file.endswith(".jsonl"):
            continue

        input_full_path = os.path.join(input_dir, input_file)
        process_filtering(input_file=input_full_path, output_base=output_base,
                          output_file=os.path.join(output_dir, input_file), logger=logger)

    return output_dir


def process_filtering(input_file: str, output_base: str, output_file: str, debug: bool = False, *, logger=None):
    logger = logger or Logger.get_logger(__name__, logdir=os.path.join(output_base, "log"))
    cleaner = Compose([
        document_filters.JSONLoader(),
        document_filters.DocumentNormalizer(),
        DiscardAdultContentJa(),
        DiscardBBSComments(),
        document_filters.DiscardAds(),
        DiscardDiscriminationContentJa(),
        RemoveRepetition(),
        NewLineSentenceTokenizer(),
        RemoveHeadTailWhitespaceTokenizer(),
        RemoveIncompleteSentence(),
        MergeTokens(delimiter="\n"),
        NewLineSentenceTokenizer(),
        DiscardSpecialCharactersJa(),
        MergeTokens(delimiter="\n"),
        NewLineSentenceTokenizer(),
        RemoveOnewordNumber(),
        MergeTokens(delimiter="\n"),
        document_filters.MaskPersonalInformation(),
        document_filters.JSONDumper(dump_reason=True),
    ])

    input_doc_iter = (Document(line) for line in lib.readlines(input_file))
    num_jobs = os.environ.get("NUM_WORKER", NUM_WORKER)
    with hojichar.Parallel(cleaner, num_jobs=num_jobs) as filter:
        out_doc_iter = filter.imap_apply(input_doc_iter)

        with open(output_file, "w") as writer:
            for result in out_doc_iter:
                try:
                    if not result.is_rejected:
                        writer.write(result.text + "\n")
                except Exception as e:
                    logger.error(f"Error processing document: {e}")

    if debug:
        os.makedirs(os.path.join(output_base, "stat", "filtering"), exist_ok=True)
        input_file_prefix = os.path.splitext(os.path.basename(input_file))[0]
        with open(os.path.join(output_base, f"{input_file_prefix}.jsonl"), "w") as writer:
            writer.write(json.dumps(cleaner.statistics, ensure_ascii=False) + "\n")

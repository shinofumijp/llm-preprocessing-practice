import hojichar
from hojichar import document_filters, Compose, Document

import os
import json
import logging
import multiprocessing

from preprocessing.filters.token_filters import RemoveIncompleteSentence, RemoveHeadTailWhitespaceTokenizer, DiscardSpecialCharactersJa, RemoveOnewordNumber
from preprocessing.filters.document_filters import DiscardAdultContentJa, DiscardBBSComments, DiscardDiscriminationContentJa, RemoveRepetition, NewLineSentenceTokenizer, MergeTokens
from preprocessing.dedup.dedup import url_dedup
import preprocessing.lib as lib


NUM_WORKER = (multiprocessing.cpu_count() - 1)


def __output_dir_after_url_dedup(base: str):
    return os.path.join(base, "url_dedup")


def __output_dir_after_filtering(base: str):
    return os.path.join(base, "filtering")


def __makedirs_for_output(input_file: str, output_base: str):
    input_file_prefix = os.path.splitext(os.path.basename(input_file))[0]
    output_base_for_file: str = os.path.join(output_base, input_file_prefix)
    os.makedirs(output_base_for_file, exist_ok=True)
    return output_base_for_file


def execute_url_dedup(input_dir: str, output_base: str) -> str:
    output_dir = __output_dir_after_url_dedup(output_base)
    os.makedirs(output_dir, exist_ok=True)

    for input_file in os.listdir(input_dir):
        if not input_file.endswith(".jsonl"):
            continue

        input_full_path = os.path.join(input_dir, input_file)
        output_base_for_input = __makedirs_for_output(input_full_path, output_base)
        url_dedup(input_file=input_full_path, output_base=output_base_for_input,
                  output_file=os.path.join(output_dir, input_file))

    return output_dir


def execute_filtering(input_dir: str, output_base: str) -> list[str]:
    output_dir = __output_dir_after_filtering(output_base)
    os.makedirs(output_dir, exist_ok=True)

    for input_file in os.listdir(input_dir):
        if not input_file.endswith(".jsonl"):
            continue

        input_full_path = os.path.join(input_dir, input_file)
        output_base_for_input = __makedirs_for_output(input_full_path, output_base)
        process_filtering(input_file=input_full_path, output_base=output_base_for_input,
                          output_file=os.path.join(output_dir, input_file))

    return output_dir


def process_filtering(input_file: str, output_base: str, output_file: str, debug: bool = False):
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
                    print(f"Error processing document: {e}")
                    logging.error(f"Error processing document: {e}")

    if debug:
        with open(os.path.join(output_base, "stat.jsonl"), "w") as writer:
            writer.write(json.dumps(cleaner.statistics, ensure_ascii=False) + "\n")

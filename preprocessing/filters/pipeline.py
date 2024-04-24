from hojichar import document_filters, Compose, Document

import os
import json


from preprocessing.filters.token_filters import RemoveIncompleteSentence, RemoveHeadTailWhitespaceTokenizer, DiscardSpecialCharactersJa, RemoveOnewordNumber
from preprocessing.filters.document_filters import DiscardAdultContentJa, DiscardBBSComments, DiscardDiscriminationContentJa, RemoveRepetition, NewLineSentenceTokenizer, MergeTokens
from preprocessing.dedup.dedup import url_dedup


def __makedirs_for_output(input_file: str, output_base: str):
    input_file_prefix = os.path.splitext(os.path.basename(input_file))[0]
    output_base_for_file: str = os.path.join(output_base, input_file_prefix)
    os.makedirs(output_base_for_file, exist_ok=True)
    return output_base_for_file


def execute_filtering(file_lines: dict[str, list[str]], input_dir: str, output_base: str) -> list[str]:
    stats, url_set = [], set()
    for input_file, lines in file_lines.items():
        input_full_path = os.path.join(input_dir, input_file)
        output_base_for_input = __makedirs_for_output(input_full_path, output_base)
        lines = url_dedup(lines, url_set, output_base_for_input, stats)
        file_lines[input_file] = lines

    stats = []
    for input_file, lines in file_lines.items():
        input_full_path = os.path.join(input_dir, input_file)
        output_base_for_input = __makedirs_for_output(input_full_path, output_base)
        lines = process_filtering(lines, output_base_for_input, stats)
        file_lines[input_file] = lines

    with open(os.path.join(output_base, "result.filtering.jsonl"), "w") as writer:
        for _, lines in file_lines.items():
            for line in lines:
                writer.write(line + "\n")

    with open(os.path.join(output_base, "stat.filtering.jsonl"), "w") as writer:
        for stat in stats:
            writer.write(json.dumps(stat, ensure_ascii=False) + "\n")

    return file_lines.items()


def process_filtering(lines: list[str], output_base: str, stats: list[dict]):
    remained_lines = []
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

    with open(os.path.join(output_base, "rejected.jsonl"), "w") as rejected:
        with open(os.path.join(output_base, "result.jsonl"), "w") as writer:
            for line in lines:
                result = cleaner.apply(Document(line))
                if result.is_rejected:
                    rejected.write(result.text + "\n")
                else:
                    writer.write(result.text + "\n")
                    remained_lines.append(result.text)

    with open(os.path.join(output_base, "stat.jsonl"), "w") as writer:
        writer.write(json.dumps(cleaner.statistics, ensure_ascii=False) + "\n")

    stats.append(cleaner.statistics)

    return remained_lines

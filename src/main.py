import argparse
import json
from hojichar import document_filters, deduplication, Compose, Document
from pprint import pprint


def main(input_file: str, output_file: str):
    cleaner = Compose([
        document_filters.JSONLoader(),
        document_filters.DocumentNormalizer(),
        document_filters.DiscardAdultContentJa(skip_rejected=False),
        document_filters.DiscardBBSComments(skip_rejected=False),
        document_filters.DiscardAds(skip_rejected=False),
        document_filters.DiscardViolenceContentJa(),
        document_filters.DiscardDiscriminationContentJa(),
        document_filters.DocumentLengthFilter(),
        document_filters.AcceptJapanese(),
        document_filters.DiscardRareKuten(),
        document_filters.HeaderFooterTagsRemover(),
        document_filters.MaskPersonalInformation(skip_rejected=False),
        document_filters.JSONDumper(dump_reason=True),
    ])

    with open(input_file) as fp:
        lines = fp.readlines()

    with open(output_file, "w") as writer:
        for line in lines:
            result = cleaner.apply(Document(line))
            writer.write(result.text + "\n")

    print('-- start dedup--')
    cleaner = Compose([
        document_filters.JSONLoader(),
        deduplication.GenerateDedupLSH(),
        deduplication.LSHDeduplicator(
            online_dedup=True,
            store_blacklist=True
        ),
        document_filters.JSONDumper()
    ])

    pprint(cleaner.statistics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--input_file', type=str, help='The input file containing documents to process')
    parser.add_argument('--output_file', default="./output/result.json", type=str,
                        help='The output file to save processed documents')
    args = parser.parse_args()
    main(input_file=args.input_file, output_file=args.output_file)

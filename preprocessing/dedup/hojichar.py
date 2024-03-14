import os
import json

from hojichar import deduplication, document_filters, Compose, Document


def exec_deduplication(lines: list[str], output_base: str):
    cleaner = Compose([
        document_filters.JSONLoader(),
        deduplication.GenerateDedupLSH(),
        deduplication.LSHDeduplicator(
            online_dedup=True,
            store_blacklist=True
        ),
        document_filters.JSONDumper()
    ])

    with open(os.path.join(output_base, "result.dedup.jsonl"), "w") as writer:
        with open(os.path.join(output_base, "rejected.dedup.jsonl"), "w") as rejected:
            for line in lines:
                result = cleaner.apply(Document(line))
                if result.is_rejected:
                    rejected.write(result.text + "\n")
                else:
                    writer.write(result.text + "\n")

    with open(os.path.join(output_base, "stat.dedup.json"), "w") as writer:
        writer.write(json.dumps(cleaner.statistics, ensure_ascii=False))

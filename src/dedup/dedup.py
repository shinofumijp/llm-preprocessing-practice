from hojichar import Compose, document_filters

import os
import uuid
import redis
import json

from src.dedup.minhash import deduplicate_documents, create_minhash_lsh, create_minhash_index
from src.filters.document_filters import JSONHTMLLoader, DeduplicationByURL
from src.models.document import DocumentFromHTML


def exec_deduplication(file_lines: dict[str, list[str]], output_base: str) -> list[str]:
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", 6379)
    redis_db = os.environ.get("REDIS_DB", 0)
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
    docs: dict[str, str] = {}
    for _, lines in file_lines.items():
        for line in lines:
            data = json.loads(line)
            text = str(data["text"])
            key = uuid.uuid4()
            r.set(str(key), text)
            docs[key] = text

    lsh = create_minhash_lsh(storage_config={"type": "redis", "redis":  {
                             'host': redis_host, 'port': redis_port, 'db': redis_db}})
    create_minhash_index(lsh, docs)
    deduplicated: dict[str, str] = deduplicate_documents(docs, lsh)

    with open(os.path.join(output_base, "result.dedup.jsonl"), "w") as writer:
        for text in deduplicated.values():
            writer.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    return deduplicated.values()


def url_dedup(lines: list[str], url_set: set[str], output_base: str, stats: list[dict]) -> list[str]:
    cleaner = Compose([
        JSONHTMLLoader(),
        DeduplicationByURL(url_set=url_set),
        document_filters.JSONDumper(),
    ])

    remained_lines = []
    with open(os.path.join(output_base, "rejected.url_dedup.jsonl"), "w") as rejected:
        with open(os.path.join(output_base, "result.url_dedup.jsonl"), "w") as writer:
            for line in lines:
                result = cleaner.apply(DocumentFromHTML(line))
                if result.is_rejected:
                    rejected.write(result.text + "\n")
                else:
                    writer.write(result.text + "\n")
                    remained_lines.append(result.text)

    with open(os.path.join(output_base, "stat.url_dedup.jsonl"), "w") as writer:
        writer.write(json.dumps(cleaner.statistics, ensure_ascii=False) + "\n")

    stats.append(cleaner.statistics)

    return remained_lines

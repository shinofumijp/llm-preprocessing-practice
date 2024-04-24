from time import sleep
import time
from hojichar import Compose, document_filters

import os
import redis
import json
import logging
import random
import string

from preprocessing.dedup.minhash import deduplicate_documents, create_minhash_lsh, create_minhash_index
from preprocessing.filters.document_filters import JSONHTMLLoader, DeduplicationByURL
from preprocessing.models.document import DocumentFromHTML
from preprocessing.dedup import worker


NUM_WORKER = 10


def exec_deduplication(file_lines: dict[str, list[str]], output_base: str, input_dir: str, basename: str = "") -> list[str]:
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", 6379)
    redis_db = os.environ.get("REDIS_DB", 0)
    r = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)

    r.set("dedup_status", "processing")
    basename: str = basename if basename else ''.join(random.choice(string.ascii_lowercase)
                                                      for _ in range(11))
    try:
        # Queue files to be processed
        for filename in file_lines.keys():
            logging.info(f"Queueing {filename}")
            r.rpush("dedup_files.before_processing", filename)

        starttime = time.time()
        for worker_id in range(NUM_WORKER):
            logging.info(f"Starting worker {worker_id}")
            worker.evoke_worker(worker_id, input_dir, basename)

        while r.llen("dedup_files.before_processing") > 0 or r.hlen("dedup_files.processing") > 0:
            logging.info(f"Waiting for processing to finish. {r.llen('dedup_files.before_processing')} files left")
            sleep(10)
        endtime = time.time()
        logging.info(f"Processing time to create minhash index : {endtime - starttime}")

        starttime = time.time()
        docs: dict[str, str] = r.hgetall("dedup.docs")

        lsh = create_minhash_lsh(storage_config={"type": "redis", "basename": basename.encode('utf8'), "redis":  {
            'host': redis_host, 'port': redis_port, 'db': redis_db}})
        deduplicated: dict[str, str] = deduplicate_documents(docs, lsh, redis=r)
        endtime = time.time()
        logging.info(f"Processing time to dedup: {endtime - starttime}")

        with open(os.path.join(output_base, "result.dedup.jsonl"), "w") as writer:
            for text in deduplicated.values():
                writer.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

        return deduplicated.values()
    except Exception as e:
        r.delete("dedup_files.before_processing")
        raise e


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

from time import sleep
import time
import hojichar
from hojichar import Compose, document_filters

import os
import redis
import json
import random
import string
import multiprocessing

from preprocessing.dedup.minhash import deduplicate_documents, create_minhash_lsh, create_minhash_index
from preprocessing.filters.document_filters import JSONHTMLLoader, DeduplicationByURL
from preprocessing.models.document import DocumentFromHTML
from preprocessing.dedup import worker
from multiprocessing import Pool, cpu_count
import preprocessing.lib as lib


NUM_WORKER = (multiprocessing.cpu_count() - 2)


def exec_deduplication(output_base: str, input_dir: str, basename: str = "", *, logger=None) -> list[str]:
    logger = logger or lib.Logger.get_logger(__name__, logdir=os.path.join(os.getcwd(), "log"))

    log_dir = os.path.join(output_base, "log")

    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", 6379)
    redis_db = os.environ.get("REDIS_DB", 0)
    r = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)

    r.set("dedup_status", "processing")
    basename: str = basename if basename else ''.join(random.choice(string.ascii_lowercase)
                                                      for _ in range(11))
    try:
        # Queue files to be processed
        for filename in os.listdir(input_dir):
            if not filename.endswith(".jsonl"):
                continue

            logger.info(f"Queueing {filename}")
            r.rpush("dedup_files.before_processing", filename)

        starttime = time.time()
        num_jobs = os.environ.get("NUM_WORKER", NUM_WORKER)
        for worker_id in range(num_jobs):
            logger.info(f"Starting worker {worker_id}")
            worker.evoke_worker(worker_id=worker_id, input_dir=input_dir, log_dir=log_dir, basename=basename)

        while r.llen("dedup_files.before_processing") > 0 or r.hlen("dedup_files.processing") > 0:
            logger.info(f"Waiting for processing to finish. {r.llen('dedup_files.before_processing')} files left")
            sleep(10)
        endtime = time.time()
        logger.info(f"Processing time to create minhash index : {endtime - starttime}")

        starttime = time.time()
        docs: dict[str, str] = r.hgetall("dedup.docs")

        lsh = create_minhash_lsh(storage_config={"type": "redis", "basename": basename.encode('utf8'), "redis":  {
            'host': redis_host, 'port': redis_port, 'db': redis_db}})
        deduplicated: list[str] = deduplicate_documents(docs, lsh, redis=r, logger=logger)
        endtime = time.time()
        logger.info(f"Processing time to dedup: {endtime - starttime}")

        output_dir = os.path.join(output_base, "minhash_dedup")
        os.makedirs(output_dir, exist_ok=True)
        write_dedup_result(texts=deduplicated, output_dir=output_dir)

        return output_dir
    except Exception as e:
        r.delete("dedup_files.before_processing")
        raise e


def __write_to_file(args):
    texts, file_id, output_dir, order = args
    with open(os.path.join(output_dir, file_id.zfill(order) + ".jsonl"), "w") as writer:
        for text in texts:
            writer.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def write_dedup_result(texts: list[str], output_dir: str):
    num_jobs = os.environ.get("NUM_WORKER", NUM_WORKER)
    num_items = len(texts)
    num_lines_per_file = 10_000
    order = (num_items // num_lines_per_file) + 1

    # Split deduplicated.values() into chunks for each process
    chunks = [list() for _ in range(num_jobs)]
    for i, text in enumerate(texts):
        chunks[i % num_jobs].append(text)

    # Prepare arguments for each process
    args = [(chunks[i], str(i), output_dir, order) for i in range(num_jobs)]

    # Use multiprocessing Pool to write to files in parallel
    with Pool(num_jobs) as p:
        p.map(__write_to_file, args)


def url_dedup(input_file: str, output_base: str, output_file: str, debug: bool = False, *, logger=None) -> list[str]:
    logger = logger or lib.Logger.get_logger(__name__, logdir=os.path.join(os.getcwd(), "log"))
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", 6379)
    redis_db = os.environ.get("REDIS_DB", 0)
    cleaner = Compose([
        JSONHTMLLoader(),
        DeduplicationByURL(redis_host=redis_host, redis_port=redis_port, redis_db=redis_db, basename=output_base),
        document_filters.JSONDumper(),
    ])

    input_doc_iter = (DocumentFromHTML(line) for line in lib.readlines(input_file))
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
        os.makedirs(os.path.join(output_base, "stat", "url_dedup"), exist_ok=True)
        input_file_prefix = os.path.splitext(os.path.basename(input_file))[0]
        with open(os.path.join(output_base, f"{input_file_prefix}.jsonl"), "w") as writer:
            writer.write(json.dumps(cleaner.statistics, ensure_ascii=False) + "\n")

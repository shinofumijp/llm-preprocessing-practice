import argparse
import os
import subprocess
import uuid
import json
import logging

import redis

from preprocessing.dedup.minhash import create_minhash_index, create_minhash_lsh
from preprocessing import ROOT_PATH

SCRIPT_PATH = os.path.join(ROOT_PATH, "scripts")


def __readlines(input_file: str):
    with open(input_file, "r", encoding="utf-8") as fp:
        for line in fp:
            yield line


def evoke_worker(worker_id: int, input_dir: str, basename: str) -> None:
    subprocess.Popen([os.path.join(SCRIPT_PATH, "throw_job.sh"), ROOT_PATH, str(worker_id), input_dir, basename])


def run(worker_id: int, input_dir: str, basename: str):
    logging.info(f"Worker {worker_id} started")
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", 6379)
    redis_db = os.environ.get("REDIS_DB", 0)
    r = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
    lsh = create_minhash_lsh(storage_config={"type": "redis", "basename": basename.encode('utf8'), "redis": {
        'host': redis_host, 'port': redis_port, 'db': redis_db}})

    while True:
        try:
            with r.lock("lock.dedup_files.before_processing", blocking_timeout=10):
                filename = r.lpop("dedup_files.before_processing")
                logging.info(f"Worker {worker_id} processing {filename}")
        except redis.exceptions.LockError:
            logging.info(f"Worker {worker_id} failed to acquire lock")
            continue

        if filename is None:
            break

        try:
            r.hset("dedup_files.processing", filename, 1)
            lines = __readlines(os.path.join(input_dir, filename))

            docs: dict[str, str] = {}
            for line in lines:
                doc_id = str(uuid.uuid4())
                docs[doc_id] = str(json.loads(line)["text"])
                r.hset("dedup.docs", doc_id, docs[doc_id])

            create_minhash_index(lsh, docs, redis=r)
            r.hdel("dedup_files.processing", filename)
        except Exception as e:
            r.rpush("dedup_files.error", filename)
            r.hdel("dedup_files.processing", filename)
            r.set("dedup_status", "error")


def main():
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--worker_id', type=int, help='The worker id', required=True)
    parser.add_argument('--input_dir', type=str,
                        help='The input directory containing documents to process', required=True)
    parser.add_argument('--basename', type=str,
                        help='The basename to use for the redis keys', required=True)
    args = parser.parse_args()
    run(args.worker_id, args.input_dir, args.basename)


if __name__ == "__main__":
    main()

from datasketch import MinHash, MinHashLSH
from transformers import GPT2Tokenizer
import unicodedata
import re
import os

from preprocessing.lib import Logger
from preprocessing.models.datastructures.unionfind import UnionFind


def normalize_document(text: str):
    """
    文書の正規化関数
    """
    # Unicode正規化
    text = unicodedata.normalize('NFKC', text)
    # 句読点の除去
    text = re.sub(r'[^\w\s]', '', text)
    # 小文字化
    text = text.lower()
    # 空白の正規化
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def create_minhash(text: str, n: int = 5, num_perm: int = 128, redis=None, doc_id: str = ""):
    """
    文書をトークン化し、MinHashオブジェクトを生成する関数
    """
    if redis is not None and doc_id != "":
        minhash = redis.lrange(f"dedup_files.minhash.{doc_id}", 0, -1)
        if minhash:
            return MinHash(num_perm, hashvalues=list(minhash))

    n_grams = normalize_and_tokenize(text, n)
    minhash = MinHash(num_perm=num_perm)
    for n_gram in n_grams:
        minhash.update(n_gram.encode('utf8'))

    if redis:
        redis.rpush(f"dedup_files.minhash.{doc_id}", *[str(v) for v in minhash.hashvalues])

    return minhash


def tokenize_docs(text: str, n=5):
    """
    文書をトークン化し、MinHashオブジェクトを生成する関数
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.tokenize(text)

    return set([' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])


def normalize_and_tokenize(text: str, n: int = 5):
    normalized_text = normalize_document(text)
    return tokenize_docs(normalized_text, n)


def process_batch(batch: dict[str, str], lsh, n=5, num_perm=128, *, redis=None, logger=None):
    logger = logger or Logger.get_logger(__name__, logdir=os.path.join(os.getcwd(), "log"))
    minhashes = {doc_id: create_minhash(text, n, num_perm, redis, doc_id) for doc_id, text in batch.items()}
    with lsh.insertion_session() as session:
        for doc_id, minhash in minhashes.items():
            try:
                session.insert(doc_id, minhash)
            except ValueError as e:
                logger.error(f"Error in inserting {doc_id}")
                logger.error(e)


def create_minhash_index(lsh: MinHashLSH, documents: dict[str, str], batch_size=1000, redis=None):
    doc_ids = list(documents.keys())
    for i in range(0, len(documents), batch_size):
        batch = {doc_id: documents[doc_id] for doc_id in doc_ids[i:i+batch_size]}
        process_batch(batch, lsh, redis=redis)
    return lsh


def create_minhash_lsh(threshold=0.9, num_perm=128, storage_config=None):
    if storage_config is None:
        storage_config = {
            'type': os.environ.get('LSH_STORAGE_CONFIG_TYPE', 'dict'),
            'name': os.environ.get('LSH_STORAGE_CONFIG_NAME'),
        }
    return MinHashLSH(threshold=threshold, num_perm=num_perm, storage_config=storage_config)


def deduplicate_documents(documents: dict[str, str], lsh: MinHashLSH, n: int = 5, num_perm: int = 128, *, redis=None, logger=None) -> list[str]:
    logger = logger or Logger.get_logger(__name__, logdir=os.path.join(os.getcwd(), "log"))

    uf = UnionFind(list(documents.keys()))
    for idx, doc in documents.items():
        m = create_minhash(doc, n, num_perm, redis, idx)
        try:
            result = lsh.query(m)
        except ValueError as e:
            logger.error(f"Error in querying {idx}")
            logger.error(e)
        for res in result:
            try:
                uf.union(idx, res)
            except KeyError:
                logger.error(f"Error in union {idx} and {res}")
    clusters = uf.groups()

    deduplicated_docs = {}
    for cluster in clusters:
        k = max(cluster, key=lambda x: x)
        deduplicated_docs.update({k: documents[k]})

    return deduplicated_docs.values()

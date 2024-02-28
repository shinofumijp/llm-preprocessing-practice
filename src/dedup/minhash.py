from datasketch import MinHash, MinHashLSH
from transformers import GPT2Tokenizer
import unicodedata
import re
import os

from src.models.datastructures.unionfind import UnionFind


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


def create_minhash(text: str, n: int = 5, num_perm: int = 128):
    """
    文書をトークン化し、MinHashオブジェクトを生成する関数
    """
    n_grams = normalize_and_tokenize(text, n)
    minhash = MinHash(num_perm=num_perm)
    for n_gram in n_grams:
        minhash.update(n_gram.encode('utf8'))
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


def process_batch(batch: dict[str, str], lsh, n=5, num_perm=128):
    minhashes = {k: create_minhash(text, n, num_perm) for k, text in batch.items()}
    with lsh.insertion_session() as session:
        for doc_id, minhash in minhashes.items():
            session.insert(doc_id, minhash)


def create_minhash_index(lsh: MinHashLSH, documents: dict[str, str], batch_size=1000):
    keys = list(documents.keys())
    for i in range(0, len(documents), batch_size):
        batch = {k: documents[k] for k in keys[i:i+batch_size]}
        process_batch(batch, lsh)
    return lsh


def create_minhash_lsh(threshold=0.9, num_perm=128, storage_config=None):
    if storage_config is None:
        storage_config = {
            'type': os.environ.get('LSH_STORAGE_CONFIG_TYPE', 'dict'),
            'name': os.environ.get('LSH_STORAGE_CONFIG_NAME'),
        }
    return MinHashLSH(threshold=threshold, num_perm=num_perm, storage_config=storage_config)


def deduplicate_documents(documents: dict[str, str], lsh: MinHashLSH, n: int = 5, num_perm: int = 128):
    uf = UnionFind(documents.keys())
    for idx, doc in documents.items():
        m = create_minhash(doc, n, num_perm)
        result = lsh.query(m)
        for res in result:
            uf.union(idx, res)
    clusters = uf.groups()

    deduplicated_docs = {}
    for cluster in clusters:
        k = max(cluster, key=lambda x: x)
        deduplicated_docs.update({k: documents[k]})

    return deduplicated_docs

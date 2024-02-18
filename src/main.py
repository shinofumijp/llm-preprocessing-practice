import argparse
from datetime import datetime
import re
import nltk
import json
import hashlib
from hojichar import document_filters, tokenization, deduplication, Compose, Document, Filter, TokenFilter, Token
from fugashi import Tagger
from typing import Any, Union
from os import PathLike
import os

tagger = Tagger('-Owakati')


def __load_and_write(lines, output_base: str, token_filter: Filter):
    removed_tokens: dict[str, set[str]] = {}
    cleaner = Compose([
        document_filters.JSONLoader(),
        document_filters.DocumentNormalizer(),
        NewLineSentenceTokenizer(),
        RemoveHeadTailWhitespaceTokenizer(),
        token_filter,
        MergeTokens(delimiter="\n", before_merge_callback=BeforeMergeTokenCallback(
            type(token_filter).__name__, removed_tokens)),
        document_filters.JSONDumper(dump_reason=True),
    ])

    for line in lines:
        cleaner.apply(Document(line))

    for filter, remove_token in removed_tokens.items():
        with open(os.path.join(output_base, "filtered." + filter + ".txt"), "w") as writer:
            for token in remove_token:
                writer.write(token + "\n")


def process_file(input_file: str, output_base: str, remained_lines: list[str], stats):
    input_file_prefix = os.path.splitext(os.path.basename(input_file))[0]
    output_base: str = os.path.join(output_base, input_file_prefix)
    os.makedirs(output_base, exist_ok=True)

    with open(input_file) as fp:
        lines = fp.readlines()

    __load_and_write(lines, output_base, RemoveIncompleteSentence())
    __load_and_write(lines, output_base, DiscardSpecialCharactersJa())
    __load_and_write(lines, output_base, RemoveOnewordNumber())

    cleaner = Compose([
        document_filters.JSONLoader(),
        # DeduplicationByURL(),
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


def main(input_dir: str, output_dir: str, dedup: bool = False):
    start = datetime.now()
    output_base = os.path.join(output_dir, start.strftime("%Y%m%d%H%M%S"))
    os.makedirs(output_base, exist_ok=True)

    remained_lines = []
    stats = []
    [process_file(os.path.join(input_dir, input_file), output_base, remained_lines, stats)
        for input_file in os.listdir(input_dir) if input_file.endswith(".jsonl")]

    with open(os.path.join(output_base, "result.jsonl"), "w") as writer:
        for line in remained_lines:
            writer.write(line + "\n")

    with open(os.path.join(output_base, "stats.jsonl"), "w") as writer:
        for stat in stats:
            writer.write(json.dumps(stat, ensure_ascii=False) + "\n")

    if dedup:
        cleaner = Compose([
            document_filters.JSONLoader(),
            deduplication.GenerateDedupLSH(),
            deduplication.LSHDeduplicator(
                online_dedup=True,
                store_blacklist=True
            ),
            document_filters.JSONDumper()
        ])
        with open(os.path.join(output_base, "dedup.jsonl"), "w") as writer:
            for line in remained_lines:
                result = cleaner.apply(Document(line))
                writer.write(result.text + "\n")

        with open(os.path.join(output_base, "stat.dedup.json"), "w") as writer:
            writer.write(json.dumps(cleaner.statistics, ensure_ascii=False))


class DeduplicationByURL(Filter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url_set = set()

    def apply(self, document: Document) -> Document:
        if document.url in self.url_set:
            document.is_rejected = True
        else:
            self.url_set.add(document.url)
        return document


class DiscardBBSComments(Filter):
    def __init__(self, threshold: float = 0.1, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.threshold = threshold
        self.keyword_pat = re.compile(
            r"\d{4}[年\.\-\/][\ ]*\d{1,2}[月\.\-\/][\ ]*\d{1,2}[日]*|コメント|SOLD OUT|レビュー|投稿|ページ|\([月火水木金土日]\)|質問|\d+話|楽天市場|-"  # noqa
        )

    def apply(self, doc: Document) -> Document:
        """
        >>> DiscardBBSComments().apply(Document("楽天市場 質問 投稿 コメント レビュー "*3)).is_rejected
        True

        >>> DiscardBBSComments().apply(Document("鏡餅")).is_rejected
        False
        """
        bbs_factor = self.keyword_pat.findall(doc.text)
        total_words = len(tagger.parse(doc.text).split())
        if total_words > 0 and len(bbs_factor) / total_words > self.threshold:
            doc.is_rejected = True
        return doc


class RemoveRepetition(Filter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.duplicate_line_fraction = 0.5
        self.duplicate_line_character_fraction = 0.5

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def apply(self, document: Document) -> Document:
        if not document.text:
            return document

        document.tokens = [Token(text) for text in document.text.split("\n")]

        line_count = 0
        dup_line = 0
        dup_line_chars = 0
        visit_lines = {}
        for token in document.tokens:
            line_hash = self._hash_text(token.text)
            if line_hash in visit_lines:
                dup_line += 1
                dup_line_chars += len(token.text)
                token.is_rejected = True
            visit_lines[line_hash] = True

            line_count += 1

        if line_count and float(dup_line) / line_count > self.duplicate_line_fraction:
            document.is_rejected = True
            return document

        if document.text and float(dup_line_chars) / len(document.text) > self.duplicate_line_character_fraction:
            document.is_rejected = True
            return document

        top_ngram_character_fractions = [
            (2, 0.2),
            (3, 0.18),
            (4, 0.16),
        ]
        for ngram, threshold in top_ngram_character_fractions:
            word_list = tagger.parse(document.text).split()
            bgs = nltk.ngrams(word_list, ngram)
            fdist = nltk.FreqDist(bgs)
            for word_list, repeat in fdist.items():
                char_count = sum([len(word) for word in word_list])
                if char_count * (repeat - 1) / len(document.text) > threshold:
                    document.is_rejected = True
                    return document

        duplicate_ngram_character_fractions = [
            (5, 0.15),
            (6, 0.14),
            (7, 0.13),
            (8, 0.12),
            (9, 0.11),
            (10, 0.10),
        ]
        for ngram, threshold in duplicate_ngram_character_fractions:
            fdist = {}
            word_list = tagger.parse(document.text).split()
            mark = [0] * len(word_list)
            for i in range(len(word_list) - ngram + 1):
                bag = tuple(word_list[i: i + ngram])
                if bag in fdist:
                    for j in range(i, i + ngram):
                        mark[j] = len(word_list[j])
                    fdist[bag] += 1
                else:
                    fdist[bag] = 1

            if sum(mark) / float(len(document.text)) > threshold:
                document.is_rejected = True
                return document

        document.text = "\n".join([token.text for token in document.tokens if not token.is_rejected])

        return document


class RemoveIncompleteSentence(TokenFilter):
    """
    文末が句点で終わっていないトークンを破棄します。
    前処理として文をSentenceで分割する必要があります。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply(self, token: Token) -> Document:
        if not token.text.endswith(("、", "。", "」", "』", "】", ")", ".", "？", "！", "!", "?", ",", ".")):
            token.is_rejected = True

        return token


class NewLineSentenceTokenizer(Filter):
    def apply(self, document: Document) -> Document:
        tokens = self.tokenize(document.text)
        document.set_tokens(tokens)
        return document

    def tokenize(self, text: str) -> list[str]:
        return text.split("\n")


class RemoveHeadTailWhitespaceTokenizer(TokenFilter):
    def apply(self, token: Token) -> Token:
        token.text = token.text.strip()
        token.text = " ".join(token.text.strip().split())
        return token


class RemoveOnewordNumber(TokenFilter):
    def __init__(self, date_pattern: re.Pattern = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.date_pattern = self._date_pattern() if date_pattern is None else date_pattern

    def _date_pattern(self) -> str:
        return re.compile(r'(\d{2,4}([-年/])\d{1,2}([-月/])\d{1,2}日?)|(\d{2,4}([-年/])\d{1,2}([-月])?)|(\d{1,2}([-月/])\d{1,2}日?).{0,8}$')

    def apply(self, token: Token) -> Token:
        text = token.text
        word_count = len(tagger.parse(text).split())
        if word_count <= 1:
            token.is_rejected = True
        elif text.isdigit() or text.isdecimal() or text.isnumeric():
            token.is_rejected = True
        elif self.date_pattern.match(text):
            token.is_rejected = True

        return token


class MergeTokens(Filter):
    """
    Merger の実装例です.
    破棄されていないトークンを結合し, Document を更新します.
    """

    def __init__(self, delimiter: str = "", before_merge_callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delimiter = delimiter
        self.before_merge_callback = before_merge_callback

    def merge(self, tokens: list[str]) -> str:
        """
        >>> MergeTokens("\n").merge(["hoo", "bar"])
        'hoo\nbar'
        """
        return self.delimiter.join(tokens)

    def apply(self, document: Document) -> Document:
        if self.before_merge_callback is not None:
            self.before_merge_callback.apply(document)

        remained_tokens = [token.text for token in document.tokens if not token.is_rejected]
        document.text = self.merge(remained_tokens)
        return document


class FilterCallback():
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def apply(self, document: Document) -> Document:
        raise NotImplementedError


class BeforeMergeTokenCallback(FilterCallback):
    def __init__(self, key, removed_tokens={}, *args, **kwargs):
        removed_tokens[key] = set()
        self.removed_tokens = removed_tokens[key]
        self.key = key

    def apply(self, document: Document) -> Document:
        for token in document.tokens:
            if token.is_rejected:
                self.removed_tokens.add(token.text)
        return document


class DiscardAdultContentJa(document_filters.NgWordsFilterJa):
    def __init__(
        self,
        dict_path: Union[str, PathLike] = document_filters.BASE_PATH / "dict/adult_keywords_ja.txt",
        threshold: float = 0.01,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(dict_path, *args, **kwargs)
        self.threshold = threshold

    def apply(self, doc: Document) -> Document:
        adult_keywords_pattern = self.keyword_pat
        matches = re.findall(adult_keywords_pattern, doc.text)
        adult_content_count = len(matches)
        total_words_count = len(tagger.parse(doc.text).split())

        if total_words_count > 0 and adult_content_count / total_words_count > self.threshold:
            doc.is_rejected = True

        return doc


class DiscardDiscriminationContentJa(document_filters.NgWordsFilterJa):
    def __init__(
        self,
        dict_path: Union[str, PathLike] = document_filters.BASE_PATH / "dict/discrimination_keywords_ja.txt",
        threshold: float = 0.01,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(dict_path, *args, **kwargs)
        self.threshold = threshold

    def apply(self, doc: Document) -> Document:
        adult_keywords_pattern = self.keyword_pat
        matches = re.findall(adult_keywords_pattern, doc.text)
        discrimination_content_count = len(matches)
        total_words_count = len(tagger.parse(doc.text).split())

        if total_words_count > 0 and discrimination_content_count / total_words_count > self.threshold:
            doc.is_rejected = True

        return doc


class DiscardSpecialCharactersJa(TokenFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.special_characters = re.compile(r"[!-/:-@[-`{-~]")

    def apply(self, token: Token) -> Document:
        special_chars_count = len(self.special_characters.findall(token.text))
        total_chars_count = len(token.text)
        if total_chars_count > 0 and special_chars_count / total_chars_count > 0.5:
            token.is_rejected = True
        return token


class DocumentFromHTML(Document):
    def __init__(self, text: str, url: str, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.url = url

    def __str__(self) -> str:
        return f"{self.url}\n{self.text}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--input_dir', type=str,
                        help='The input directory containing documents to process', required=True)
    parser.add_argument('--output_dir', type=str,
                        help='The input file containing documents to process', required=False, default="./output")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)

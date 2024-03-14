from hojichar import document_filters, Filter, Document, Token
from fugashi import Tagger

from os import PathLike
from typing import Any, Union
import re
import json
import hashlib
import nltk


from src.models.document import DocumentFromHTML


class JSONHTMLLoader(Filter):
    def __init__(self, key: str = "text", ignore: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.key = key
        self.ignore = ignore

    def apply(self, document: DocumentFromHTML) -> DocumentFromHTML:
        try:
            data = json.loads(document.text)
            document.text = str(data[self.key])
            document.url = str(data.get("url", ""))
        except Exception as e:
            if self.ignore:
                document.is_rejected = True
                return document
            else:
                raise e

        return document


class DeduplicationByURL(Filter):
    def __init__(self, url_set: set[str] = set(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url_set = url_set

    def apply(self, document: DocumentFromHTML) -> DocumentFromHTML:
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
        self.tagger = Tagger('-Owakati')

    def apply(self, doc: Document) -> Document:
        """
        >>> DiscardBBSComments().apply(Document("楽天市場 質問 投稿 コメント レビュー "*3)).is_rejected
        True

        >>> DiscardBBSComments().apply(Document("鏡餅")).is_rejected
        False
        """
        bbs_factor = self.keyword_pat.findall(doc.text)
        total_words = len(self.tagger.parse(doc.text).split())
        if total_words > 0 and len(bbs_factor) / total_words > self.threshold:
            doc.is_rejected = True
        return doc


class RemoveRepetition(Filter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.duplicate_line_fraction = 0.5
        self.duplicate_line_character_fraction = 0.5
        self.tagger = Tagger('-Owakati')

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
            word_list = self.tagger.parse(document.text).split()
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
            word_list = self.tagger.parse(document.text).split()
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


class NewLineSentenceTokenizer(Filter):
    def apply(self, document: Document) -> Document:
        tokens = self.tokenize(document.text)
        document.set_tokens(tokens)
        return document

    def tokenize(self, text: str) -> list[str]:
        return text.split("\n")


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
        self.tagger = Tagger('-Owakati')

    def apply(self, doc: Document) -> Document:
        adult_keywords_pattern = self.keyword_pat
        matches = re.findall(adult_keywords_pattern, doc.text)
        adult_content_count = len(matches)
        total_words_count = len(self.tagger.parse(doc.text).split())

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
        self.tagger = Tagger('-Owakati')

    def apply(self, doc: Document) -> Document:
        adult_keywords_pattern = self.keyword_pat
        matches = re.findall(adult_keywords_pattern, doc.text)
        discrimination_content_count = len(matches)
        total_words_count = len(self.tagger.parse(doc.text).split())

        if total_words_count > 0 and discrimination_content_count / total_words_count > self.threshold:
            doc.is_rejected = True

        return doc

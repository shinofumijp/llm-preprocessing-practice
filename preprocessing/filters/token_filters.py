from hojichar import document_filters, TokenFilter, Document, Compose, Filter, Token
from fugashi import Tagger

import os
import re

from src.filters.document_filters import NewLineSentenceTokenizer, MergeTokens, BeforeMergeTokenCallback


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


class RemoveHeadTailWhitespaceTokenizer(TokenFilter):
    def apply(self, token: Token) -> Token:
        token.text = token.text.strip()
        token.text = " ".join(token.text.strip().split())
        return token


class RemoveOnewordNumber(TokenFilter):
    def __init__(self, date_pattern: re.Pattern = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.date_pattern = self._date_pattern() if date_pattern is None else date_pattern
        self.tagger = Tagger('-Owakati')

    def _date_pattern(self) -> str:
        """
        日付のみのパターンと日付と数文字のみのパターンを削除
        例) 2024年2月2日, 2024年2月, 2月2日, 2024/2/2, 2024/2, 2/2, 2024-2-2, 2024-2, 2-2
        日付 + 数文字はHTMLの場合には検索時の補助のために用いられるもので文章になっていないことが多い
        例) 2024年2月2日 (8)
        """
        return re.compile(r'((\d{2,4}([-年/])\d{1,2}([-月/])\d{1,2}日?)|(\d{2,4}([-年/])\d{1,2}([-月])?)|(\d{1,2}([-月/])\d{1,2}日?)).{0,8}$')

    def apply(self, token: Token) -> Token:
        text = token.text
        word_count = len(self.tagger.parse(text).split())
        if word_count <= 1:
            token.is_rejected = True
        elif text.isdigit() or text.isdecimal() or text.isnumeric():
            token.is_rejected = True
        elif self.date_pattern.match(text):
            token.is_rejected = True

        return token


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


def __debug_token_filter(lines, output_base: str, token_filter: Filter):
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

from typing import Generator


def readlines(file: str) -> Generator[str, None, None]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            yield line

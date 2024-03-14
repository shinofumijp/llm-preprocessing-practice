from hojichar import Document


class DocumentFromHTML(Document):
    def __init__(self, text: str, url: str = "", *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.url = url

    def __str__(self) -> str:
        return f"{self.url}\n{self.text}"

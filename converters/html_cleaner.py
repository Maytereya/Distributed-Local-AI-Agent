from html.parser import HTMLParser
from html import unescape

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:           # Только «явный» текст
        self.parts.append(data)

    def get_data(self) -> str:
        return "".join(self.parts)

def strip_html(text: str) -> str:
    """Удаляет теги *и* переводит HTML‑сущности (&#160;, &nbsp; и т.п.)."""
    stripper = _HTMLStripper()
    stripper.feed(text)
    return unescape(stripper.get_data()).strip()
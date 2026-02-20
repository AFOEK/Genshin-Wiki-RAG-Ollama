import re

class Filters:
    def __init__(self, deny_url_regex: str, deny_text_regex: str) -> None:
        self.deny_url = re.compile(deny_url_regex) if deny_url_regex else None
        self.deny_text = re.compile(deny_text_regex) if deny_text_regex else None

    def url_allowed(self, url: str) -> bool:
        return not (self.deny_url and self.deny_url.search(url))
    
    def text_allowed(self, text: str) -> bool:
        return not(self.deny_text and self.deny_text.search(text))
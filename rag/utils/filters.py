import re

class Filters:
    def __init__(self, deny_url_regex: str | None = None, deny_text_regex: str | None = None, allow_url_regex: str | None = None) -> None:
        self.deny_url = re.compile(deny_url_regex, re.I) if deny_url_regex else None
        self.deny_text = re.compile(deny_text_regex, re.I) if deny_text_regex else None
        self.allow_url = re.compile(allow_url_regex, re.I) if allow_url_regex else None

    def url_allowed(self, url: str) -> bool:
        if self.deny_url and self.deny_url.search(url):
            return False
        if self.allow_url and not self.allow_url.search(url):
            return False
        return True
    
    def text_allowed(self, text: str) -> bool:
        return not (self.deny_text and self.deny_text.search(text))
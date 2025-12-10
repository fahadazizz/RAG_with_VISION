import re


class TextCleaner:
    
    def __init__(self, remove_special_chars: bool = False):
        self.remove_special_chars = remove_special_chars
        self._whitespace_pattern = re.compile(r'\s+')
        self._special_chars_pattern = re.compile(r'[^\w\s.,!?;:\'"()-]')
        self._multiple_newlines_pattern = re.compile(r'\n{3,}')
        self._header_footer_pattern = re.compile(
            r'^(page\s*\d+|©.*|all rights reserved.*|\d+\s*of\s*\d+).*$',
            re.MULTILINE | re.IGNORECASE
        )

    def clean(self, text: str) -> str:
        if not text:
            return ""
        
        if self.remove_special_chars:
            text = self._special_chars_pattern.sub(' ', text)
        
        text = self.normalize_whitespace(text)
        return text.strip()
    
    def normalize_whitespace(self, text: str) -> str:
        text = self._multiple_newlines_pattern.sub('\n\n', text)
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            normalized_line = self._whitespace_pattern.sub(' ', line).strip()
            normalized_lines.append(normalized_line)
        return '\n'.join(normalized_lines)
    
    def remove_headers_footers(self, text: str) -> str:
        return self._header_footer_pattern.sub('', text)
    
    def extract_sentences(self, text: str) -> list[str]:
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

from tools.utils.text_cleaner import TextCleaner


def clean_document_text(text: str, remove_headers: bool = True) -> str:
    """
    Clean document text with optional header/footer removal.
    
    Args:
        text: Raw document text
        remove_headers: Whether to remove headers/footers
        
    Returns:
        Cleaned text
    """
    cleaner = TextCleaner()
    cleaned = cleaner.clean(text)
    if remove_headers:
        cleaned = cleaner.remove_headers_footers(cleaned)
    return cleaned

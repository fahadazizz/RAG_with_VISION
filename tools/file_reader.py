from pypdf import PdfReader

def read_pdf(file_path):
    """Reads a PDF file and returns the text content.
        source: PDF 
        output: Extracted Text from pdf
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

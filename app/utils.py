# app/utils.py
import os
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_pdf_as_documents(pdf_path: str) -> List[Document]:
    """
    Reads the PDF and returns ONE Document containing full text.
    """
    reader = PdfReader(pdf_path)
    pages = []

    for page in reader.pages:
        try:
            text = page.extract_text()
        except:
            text = ""
        if text:
            pages.append(text)

    full_text = "\n\n".join(pages)
    return [Document(page_content=full_text, metadata={"source": os.path.basename(pdf_path)})]


def split_documents(documents: List[Document], chunk_size=1600, chunk_overlap=300) -> List[Document]:
    """
    Splits long content into chunks using LangChain's splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)

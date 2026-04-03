from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# Ingest PDFs
def ingest_pdfs(path:str):
    
    path = Path(path)

    # verify if the path is a file
    if path.is_file():
        # if the file is not a pdf return error
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")
        # load the pdf
        loader = PDFPlumberLoader(str(path))
    
    # in case the path is a folder path
    elif path.is_dir():
        # batch loader
        loader = DirectoryLoader(
            str(path),
            glob="**/*.pdf",
            loader_cls=PDFPlumberLoader,
            use_multithreading=True,
            show_progress=True
        )
    # raise exception in case the path does not exist
    else:
        raise ValueError(f"Path does not exist: {path}")
    
    # load the file(s) in the path
    docs = loader.load()
    
    # raise exception in case there is no information
    if not docs:
        raise ValueError(f"No PDF content found at: {path}")
    
    # # print how many files will be loaded
    # print(f"Loaded {len(set(doc.metadata['source'] for doc in docs))} file(s)")
    return docs
    

# Split the documents into chunks for embedding
def doc_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  # chunk size
        chunk_overlap=128,  # chunk overlap
        add_start_index=True,  # track index in original document
    )
    # splited documents
    all_splits = text_splitter.split_documents(docs)

    return all_splits

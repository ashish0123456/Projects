import os
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from config import DATA_DIR, CHROMA_DIR, TAGGED_FILE

class UTF8TextLoader(TextLoader):
    def __init__(self, file_path):
        super().__init__(file_path, encoding='utf-8')

def main():
    # Load books with emotions
    books_path = os.path.join(DATA_DIR, 'books_with_emotions.csv')
    books = pd.read_csv(books_path)
    books['large_thumbnail'] = books['thumbnail'].fillna('') + "&fife=w800"
    books['large_thumbnail'].replace('&fife=w800', 
                                     os.path.join(DATA_DIR, 'cover-not_found.jpg'), inplace=True)
    
    # Load & split tagged descriptions
    tagged_path = os.path.join(DATA_DIR, TAGGED_FILE)
    raw = UTF8TextLoader(tagged_path).load()

    # Use a chunk size of 0 with no overlap if you want the whole text at once
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw)

    # (Re)create Chroma directory
    os.makedirs(CHROMA_DIR, exist_ok=True)
    db_books = Chroma(embedding_function=FastEmbedEmbeddings(),
                persist_directory=CHROMA_DIR)
    
    # clear existing and initialize new collection
    db_books.reset_collection()

    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i: i + batch_size]
        db_books.add_documents(batch)

    print(f"Indexed {len(documents)} documents into {CHROMA_DIR}")

if __name__ == '__main__':
    main()
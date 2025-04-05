import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
CHROMA_DIR = os.path.join(DATA_DIR, 'chroma_db')
TAGGED_FILE = 'tagged_descriptions.txt'
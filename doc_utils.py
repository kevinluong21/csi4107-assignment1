import json
import os

from indexing import InvertedIndex

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def save_inverted_index_jsonl(inv_index, file_path):
    with open(file_path, 'w') as file:
        for term, postings in inv_index.index.items():
            file.write(json.dumps({term: postings}) + "\n")

# Load inverted index from JSONL file
def load_inverted_index_jsonl(file_path):
    inverted_index = InvertedIndex()
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                entry = json.loads(line.strip())
                for term, postings in entry.items():
                    inverted_index.index[term] = postings
    return inverted_index


from collections import defaultdict
import os

from indexing import InvertedIndex
from retrieve_and_rank import get_document_vector, process_and_save_results, rank_documents_for_query
from preprocessing import Document
from doc_utils import load_inverted_index_jsonl,load_jsonl, save_inverted_index_jsonl

#Corpus loading 
# corpus = load_jsonl('corpus_first_5.jsonl')  # 5 documents
queries = load_jsonl('queries_for_test.jsonl')  # 50 queries

# corpus = load_jsonl(r"scifact/corpus.jsonl")
# queries = load_jsonl(r"scifact/queries.jsonl")

corpus = load_jsonl('corpus.jsonl')  # all
# queries = load_jsonl('queries_for_test.jsonl')  # test queries

documents = []
i = 1

for doc in corpus:
    print(f"Loading document {i} of {len(corpus)}...")
    i += 1

    document = Document(title=doc['title'], text="", _id=doc['_id'], metadata=doc['metadata'])
    
    documents.append(document)
    
# index_file_path = "save_inv_check.jsonl"
index_file_path = "inverted_index_titles.jsonl"

if os.path.exists(index_file_path):
    # Load the existing index
    inv_index = load_inverted_index_jsonl(index_file_path)
    print("Loaded existing inverted index.")
else:
    # Create and save a new inverted index
    inv_index = InvertedIndex()
    # Add documents to inverted index
    for document in documents:
        
        terms = document.get_index_terms() 
       
        # print(document._id)
        inv_index.add_documents(document._id, terms)

    save_inverted_index_jsonl(inv_index, index_file_path)
    print("Saved new inverted index.")
    

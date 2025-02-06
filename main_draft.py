from collections import defaultdict
import os

from indexing import InvertedIndex
from retrieve_and_rank import get_document_vector, get_bm25_document_vector, process_and_save_results, rank_documents_for_query
from preprocessing import Document
from doc_utils import load_inverted_index_jsonl,load_jsonl, save_inverted_index_jsonl

#Corpus loading 
# corpus = load_jsonl('corpus_first_5.jsonl')  # 5 documents
# queries = load_jsonl('queries_for_test.jsonl')  # 50 queries

# corpus = load_jsonl(r"scifact/corpus.jsonl")
# queries = load_jsonl(r"scifact/queries.jsonl")

corpus = load_jsonl('scifact/corpus.jsonl')  # all
queries = load_jsonl('queries_for_test.jsonl')  # test queries

documents = {}
i = 1

for doc in corpus:
    print(f"Loading document {i} of {len(corpus)}...")
    i += 1

    documents[doc["_id"]] = Document(title=doc['title'], text=doc['text'], _id=doc['_id'], metadata=doc['metadata'])
    
    # documents.append(document)
    
#add the path to the inverted index (TODO: make a parameterized script)
# index_file_path = "save_inv_check.jsonl"
index_file_path = "inverted_index.jsonl"
index_file_path_titles = "inverted_index_titles.jsonl"

if os.path.exists(index_file_path):
    # Load the existing index
    inv_index = load_inverted_index_jsonl(index_file_path)
    print("Loaded existing inverted index.")
else:
    # Create and save a new inverted index
    inv_index = InvertedIndex()
    # Add documents to inverted index
    for document in documents.values():
        
        terms = document.get_index_terms() 
       
        # print(document._id)
        inv_index.add_documents(document._id, terms)

    save_inverted_index_jsonl(inv_index, index_file_path)
    print("Saved new inverted index.")
    

document_vectors = {}

# Calculate the average document length
avg_doc_length = 0

for _id, document in documents.items():
    avg_doc_length += len(document.get_index_terms())

avg_doc_length = avg_doc_length / len(documents)

for _id, document in documents.items():

    document_id = _id

    # doc_vector = get_document_vector(document_id, inv_index, len(documents))
    doc_vector = get_bm25_document_vector(document, inv_index, len(documents), avg_doc_length, delta=0.25)

    document_vectors[document_id] = doc_vector
    

top_documents_for_all_queries = []


process_and_save_results(
    queries=queries, 
    inv_index=inv_index, 
    document_vectors=document_vectors, 
    documents=documents, 
    avg_doc_length=avg_doc_length,
    output_file_name="bm25_result_for_titles_and_text.txt", 
    k1=1.0,
    b=0.5,
    delta=0.25,
    top_n=100 
)

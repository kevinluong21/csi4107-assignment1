from collections import defaultdict
import os

from indexing import InvertedIndex
from retrieve_and_rank import get_document_vector, process_and_save_results, rank_documents_for_query
from preprocessing import Document
from doc_utils import load_inverted_index_jsonl,load_jsonl, save_inverted_index_jsonl

#Corpus loading 
# corpus = load_jsonl('corpus_first_5.jsonl')  # 5 documents
# queries = load_jsonl('queries_50.jsonl')  # 50 queries

corpus = load_jsonl(r"scifact/corpus.jsonl")
queries = load_jsonl(r"scifact/queries.jsonl")

documents = []
i = 1

for doc in corpus:
    print(f"Loading document {i} of {len(corpus)}...")
    i += 1

    document = Document(title=doc['title'], text=doc['text'], _id=doc['_id'], metadata=doc['metadata'])
    
    documents.append(document)
    
#add the path to the inverted index (TODO: make a parameterized script)
# index_file_path = "save_inv_check.jsonl"
index_file_path = "inverted_index.jsonl"

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
    

document_vectors = {}

for document in documents:

    document_id = document._id

    doc_vector = get_document_vector(document_id, inv_index, len(documents))

    document_vectors[document_id] = doc_vector
    

top_documents_for_all_queries = []

# Process each query
for q in range(len(queries)):
    query_id = queries[q]['_id']
    query_text = queries[q]['text']
    
    print(f"Processing query {query_id}: {query_text}") 
    
    top_documents = rank_documents_for_query(query_text, inv_index, document_vectors, len(documents), top_n=5)
    
    # Get the top document (document with highest similarity score)
    top_doc = top_documents[0]  
    
    # Save the top document for this query
    top_documents_for_all_queries.append((query_id, top_doc[0], top_doc[1]))  # (query_id, doc_id, similarity)
    
    # Optionally, display the top documents and their similarity scores
    print(f"Top Documents for Query {query_id}:")
    for doc_id, similarity in top_documents:
        print(f"Document ID: {doc_id}, Similarity: {similarity}")
    
    # Display the top match for the current query
    print(f"Best Match for Query {query_id}: Document ID {top_doc[0]} with Similarity {top_doc[1]}")
    print("")

# After all queries, find the overall best match
best_match_overall = max(top_documents_for_all_queries, key=lambda x: x[2])  # Max by similarity score (third value in tuple)

# Display the overall best match
query_id, doc_id, similarity = best_match_overall
print(f"Overall Best Match: Query ID {query_id}, Document ID {doc_id}, Similarity {similarity}")



# process_and_save_results(
#     queries=queries, 
#     inv_index=inv_index, 
#     document_vectors=document_vectors, 
#     documents=documents, 
#     output_file_name="results.txt", 
#     top_n=2 
# )
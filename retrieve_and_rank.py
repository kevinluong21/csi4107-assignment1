from math import log, sqrt
from indexing import InvertedIndex
from preprocessing import Document, extract_index_terms #document import is needed for running some test cases


def compute_tf_idf(term_freq, doc_freq, total_docs, max_frq):
    """
    Compute the TF-IDF score for a term in a document.

    Parameters:
    - term_freq: Frequency of the term in the document.
    - doc_freq: Number of documents in the corpus containing the term.
    - total_docs: Total number of documents in the corpus.
    - max_frq: Maximum frequency of any term in the document.

    Returns:
    - tf_idf: The computed TF-IDF score for the term.
    """

    tf = term_freq / max_frq if max_frq > 0 else 0

    idf = log(total_docs / doc_freq) if doc_freq > 0 else 0
    
    tf_idf = tf * idf
    return tf_idf

# test case from the lecture
# print(compute_tf_idf(3, 50, 10000, 3))



def get_document_vector(document_id, inverted_index: InvertedIndex, total_documents):
    """
    Generate the document vector (TF-IDF values for all terms in the document).
    
    Parameters:
    - document_id: int - The ID of the document to process
    - inverted_index: InvertedIndex - The instance of the InvertedIndex class
    - total_documents: int - Total number of documents in the corpus
    
    Returns:
    - dict: A dictionary mapping terms to their TF-IDF scores
    """

    max_frq = inverted_index.get_max_term_frequency_in_doc(document_id)
    #print(f"Max frequency for document {document_id}: {max_frq}")
    
    # Create an empty dictionary to store the TF-IDF scores
    doc_vector = {}
    
    # Iterate through all terms in the inverted index
    for term, postings in inverted_index.index.items():
        #print(f"term {term} and posting {postings}")
 
        if document_id in postings:
            term_freq = postings[document_id]

            doc_freq = len(postings)  # Number of documents the term appears in
            
            # Compute the TF-IDF score for this term
            tf_idf = compute_tf_idf(term_freq, doc_freq, total_documents, max_frq)

            # Store the score in the document vector
            doc_vector[term] = tf_idf
    
    return doc_vector


#TEST CASE

index = InvertedIndex()
index.add_documents(1, {"machine": 1, "learning": 1, "ai": 1})
index.add_documents(2, {"deep": 1, "learning": 1, "ai": 1})
index.add_documents(3, {"ai": 1, "intelligence": 1})

doc_vector = get_document_vector(1, index, total_documents=3)
print(doc_vector)
#should return {'machine': 1.0986122886681098, 'learning': 0.4054651081081644, 'ai': 0.0} 

def get_query_vector(query, inverted_index, total_documents):
   
    # Tokenize the query and count term frequencies
    query_terms = extract_index_terms(query)
    max_freq = max(query_terms.values()) if query_terms else 0


    # Compute the query vector
    query_vector = {}
    for term, term_freq in query_terms.items():
        # Get document frequency from the inverted index
        postings = inverted_index.get_postings(term)
        doc_freq = len(postings)  # Number of documents containing the term

        # Compute TF-IDF
        tf_idf = compute_tf_idf(term_freq, doc_freq, total_documents, max_freq)
        query_vector[term] = tf_idf
    
    return query_vector
    
#Test
# query = "machine learning"
# total_documents = 3

# # Simulated Inverted Index
# inverted_index = InvertedIndex()
# inverted_index.add_documents(1, {"machine": 2, "learning": 1, "ai": 3})
# inverted_index.add_documents(2, {"machine": 1, "learning": 3, "deep": 2})
# inverted_index.add_documents(3, {"ai": 1, "artificial": 2, "intelligence": 1})

# query_vector = get_query_vector(query, inverted_index, total_documents)
# print("Query Vector:", query_vector)


def compute_cosine_similarity(query_vector, doc_vector):
    """
    Compute the cosine similarity between a query vector and a document vector.

    Parameters:
    - query_vector: A dictionary representing the TF-IDF weights of terms in the query.
    - doc_vector: A dictionary representing the TF-IDF weights of terms in the document.

    Returns:
    - float: The cosine similarity score between the query and the document.
    """
    # Dot product of query and document vectors
    dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) for term in query_vector)

    # Magnitude of the query vector
    query_magnitude = sqrt(sum(value**2 for value in query_vector.values()))

    # Magnitude of the document vector
    doc_magnitude = sqrt(sum(value**2 for value in doc_vector.values()))

    if query_magnitude == 0 or doc_magnitude == 0:
        return 0.0
    
    return dot_product / (query_magnitude * doc_magnitude)

# query_vector = {'machine': 1, 'learning': 1, 'robot': 1}
# doc_vector = {'machine': 1}

# similarity = compute_cosine_similarity(query_vector, doc_vector)
# print("Cosine Similarity:", similarity)



def rank_documents_for_query(query, inverted_index, document_vectors, total_documents, top_n=10):
    
    """
    Rank documents for a given query based on their similarity scores.

    Parameters:
    - query: The text of the query.
    - inverted_index: The inverted index containing term-document mappings.
    - document_vectors: Precomputed TF-IDF vectors for all documents.
    - total_documents: The total number of documents in the corpus.
    - top_n: The number of top-ranked documents to return (default is 10).

    Returns:
    - top_documents: A list of tuples containing document IDs and their similarity scores, sorted by score.
    """
    
    # Generate query vector from the query text
    query_vector = get_query_vector(query, inverted_index, total_documents)
    # print(f"qv {query_vector}")
    # Calculate similarity between the query and each document
    similarities = {}
    for doc_id, doc_vector in document_vectors.items():
        similarity = compute_cosine_similarity(query_vector, doc_vector)
        similarities[doc_id] = similarity
    
    # Sort documents based on the similarity score
    sorted_documents = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    
    # Return the top N ranked documents
    top_documents = sorted_documents[:top_n]

    return top_documents


def process_and_save_results(queries, inv_index, document_vectors, documents, output_file_name="results.txt", top_n=10):
    """
    Process queries, rank documents, and save the top results to a file in the format:
    query_id document_id score

    Parameters:
    - queries: List of query dictionaries containing '_id' and 'text'.
    - inv_index: Inverted index used for retrieving relevant documents.
    - document_vectors: Precomputed document vectors for similarity calculation.
    - documents: List of all documents in the corpus.
    - output_file_name: Name of the file to save results (default is 'results.txt').
    - top_n: Maximum number of top documents to retrieve for each query (default is 10).
    """
    top_documents_for_all_queries = []  # To store all query results for potential further processing
    
    with open(output_file_name, "w") as output_file:
        for q in range(len(queries)):
            query_id = queries[q]['_id']
            query_text = queries[q]['text']
            
            print(f"Processing query {query_id}: {query_text}")
            
            # Rank documents for the query
            top_documents = rank_documents_for_query(query_text, inv_index, document_vectors, len(documents), top_n=top_n)
            
            # Save the top results for the current query
            for rank, (doc_id, similarity) in enumerate(top_documents, start=1):
                top_documents_for_all_queries.append((query_id, doc_id, similarity))
                # Write the result to the output file
                output_file.write(f"{query_id} {doc_id} {similarity:.6f}\n")
            
            # Optionally print the top results for debugging or review
            print(f"Top Documents for Query {query_id}:")
            for rank, (doc_id, similarity) in enumerate(top_documents[:5], start=1):  # Display top 5 for debugging
                print(f"Rank {rank}: Document ID {doc_id}, Score {similarity:.6f}")
            print("")

    print(f"Results have been saved to {output_file_name}.")
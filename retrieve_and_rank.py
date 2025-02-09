from math import log, sqrt
from indexing import InvertedIndex
from preprocessing import Document, Query

def compute_bm25(total_documents, term_freq, doc_freq, doc_length, avg_doc_length, k1=1.2, b=0.75):
    """
    Computes the BM25 weighting for a given term.

    Parameters:
        - total_documents: The total number of documents in the corpus
        - term_freq: The term frequency within the document
        - doc_freq: The document frequency of a term
        - doc_length: The length of the document
        - avg_doc_length: The average length of all documents in the corpus
        - k1: BM25+ hyperparameter (default is 1.2)
        - b: BM25+ hyperparameter (default is 0.75)

    Returns:
        - weight: The BM25 weighting of the given term
    """
    weight = (term_freq * (log((total_documents - doc_freq + 0.5) / (doc_freq + 0.5)))) / ((k1 * ((1 - b) + (b * doc_length / avg_doc_length))) + term_freq)
    return weight

def compute_bm25_plus(total_documents, term_freq, doc_freq, doc_length, avg_doc_length, k1=1.2, b=0.75, delta=1):
    """
    Computes the BM25+ weighting for a given term.

    Parameters:
        - total_documents: The total number of documents in the corpus
        - term_freq: The term frequency within the document
        - doc_freq: The document frequency of a term
        - doc_length: The length of the document
        - avg_doc_length: The average length of all documents in the corpus
        - k1: BM25+ hyperparameter (default is 1.2)
        - b: BM25+ hyperparameter (default is 0.75)
        - delta: BM25+ hyperparameter (default is 1)

    Returns:
        - weight: The BM25+ weighting of the given term
    """
    weight = ((term_freq + delta) * (log((total_documents - doc_freq + 0.5) / (doc_freq + 0.5)))) / ((k1 * ((1 - b) + (b * doc_length / avg_doc_length))) + term_freq)
    return weight

def get_bm25_document_vector(document: Document, inverted_index: InvertedIndex, total_documents, avg_doc_length, k1=1.2, b=0.75, delta=1):
    """
    Tokenize each document using BM25 weighting.

    Parameters:
        - document: A Document object to tokenize
        - inverted_index: Inverted index used for retrieving relevant documents.
        - documents: List of all documents in the corpus.
        - avg_doc_length: The average document length in index terms.
        - k1: BM25+ hyperparameter (default is 1.2)
        - b: BM25+ hyperparameter (default is 0.75)
        - delta: BM25+ hyperparameter (default is 1)

    Returns:
        - doc_vector: The tokenized document as a dictionary where the key is the term and the value is the BM25+ weight
    """
    # Create an empty dictionary to store the TF-IDF scores
    doc_vector = {}

    index_terms = document.get_index_terms()
    doc_length = len(index_terms)

    for term, term_freq in index_terms.items():
        doc_freq = len(inverted_index.get_postings(term))

        weight = compute_bm25_plus(total_documents, term_freq, doc_freq, doc_length, avg_doc_length, k1=k1, b=b, delta=delta)

        doc_vector[term] = weight
    
    return doc_vector

def get_bm25_query_vector(query: Query, document: Document, inverted_index, total_documents, avg_doc_length, k1=1.2, b=0.75, delta=1):
    """
    Tokenize a query vector using BM25 weighting. Each vector is dependent on the document.

    Parameters:
        - query: A Query object
        - document: A Document object that will be used to compute the similarity against the query
        - inverted_index: Inverted index used for retrieving relevant documents.
        - documents: List of all documents in the corpus.
        - avg_doc_length: The average document length in index terms.
        - k1: BM25+ hyperparameter (default is 1.2)
        - b: BM25+ hyperparameter (default is 0.75)
        - delta: BM25+ hyperparameter (default is 1)

    Returns:
        - query_vector: The tokenized query (for the given Document object) as a dictionary where the key is the term and the value is the BM25+ weight
    """
    # Tokenize the query and count term frequencies
    query_terms = query.get_index_terms()

    index_terms = document.get_index_terms()

    doc_length = len(document)

    query_vector = {}
    for term in query_terms.keys():
        # Get document frequency from the inverted index
        doc_freq = len(inverted_index.get_postings(term))
        # Get the query term frequency in the given document
        term_freq = index_terms.get(term, 0)

        if term_freq > 0:
            weight = compute_bm25_plus(total_documents, term_freq, doc_freq, doc_length, avg_doc_length, k1=k1, b=b, delta=delta)
            query_vector[term] = weight
        else:
            query_vector[term] = 0

    return query_vector


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


def bm25_rank_documents_for_query(query: Query, inverted_index, document_vectors, documents: dict, avg_doc_length, k1=1.2, b=0.75, delta=1, top_n=100):
    """
    Using BM25 scores, rank the documents for each query.

    Parameters:
        - query: A Query object
        - inverted_index: Inverted index used for retrieving relevant documents.
        - documents: List of all documents in the corpus.
        - avg_doc_length: The average document length in index terms.
        - k1: BM25+ hyperparameter (default is 1.2)
        - b: BM25+ hyperparameter (default is 0.75)
        - delta: BM25+ hyperparameter (default is 1)
        - top_n: Maximum number of top documents to retrieve for each query (default is 100).

    Returns:
        - top_documents: The top n documents retrieved from the corpus that match the given query.
    """
    # Initialize a dictionary to store similarity scores
    similarities = {}

    # TODO: union of the query terms and the inverted index to find the documents that contain at least one query word

    for doc_id, document in documents.items():
        # Check if the inverted index contains at least one of the query words
        contains_query_word = False
        for word in query.get_index_terms():  
            if word in inverted_index.index:  # Access the index of the inverted index
                contains_query_word = True
                break

        # If the document contains query words, compute similarity
        if contains_query_word:
            query_vector = get_bm25_query_vector(query, document, inverted_index, len(documents), avg_doc_length, k1=k1, b=b, delta=delta)
            similarity = compute_cosine_similarity(query_vector, document_vectors[doc_id])
            if similarity > 0:  # Only consider documents with a non-zero similarity
                similarities[doc_id] = similarity

    # Sort the documents by similarity score in descending order
    sorted_documents = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    if not sorted_documents:
        print(f"No documents returned for query: {query}")
    top_documents = sorted_documents[:top_n]

    return top_documents

# def pseudo_relevance_loop(query: Query, documents:dict[int, Document], top_documents:list, n=2, k=3):
#     """
#     Take the top n terms of the top k documents returned by the first pass of the IR and add them to the end of the query.

#     Parameters:
#         query (str): The query as a string.
#         documents (dict): A dictionary where the document ID is the key and the Document object is the value.
#         top_documents (list): A list of top documents where each index is a tuple of the document ID and the similarity score.
#         n: The top number of words to extract from each of the top k documents. By default, 2.
#         k: The top k documents to extract terms from. By default, 3.

#     Returns:
#         query (str): The query with more terms appended to it.
#     """
#     # Split query into words and make it into a set to avoid duplicating terms to add and terms already in the query
#     query = query.get_index_terms()
#     query = set(query.keys())

#     for document in top_documents[:k]:
#         _id = document[0]
#         index_terms = documents[_id].get_index_terms()
#         # Sort the index terms by count
#         index_terms = sorted(index_terms.items(), key=lambda item: item[1], reverse=True)
#         index_terms = index_terms[:n]
#         index_terms = [term[0] for term in index_terms]

#         query = query.union(set(index_terms))

#     query = " ".join(query)
#     query = query.strip()
#     return query
  
def process_and_save_results(queries, inv_index, document_vectors, documents, avg_doc_length, output_file_name="results.txt", k1=1.2, b=0.75, delta=1, top_n=100, run_tag="run1"):
    """
    Process queries, rank documents, and save the top results in the required format.

    Output Format:
    query_id Q0 doc_id rank score run_tag

    Parameters:
    - queries: List of query dictionaries containing '_id' and 'text'.
    - inv_index: Inverted index used for retrieving relevant documents.
    - document_vectors: Precomputed document vectors for similarity calculation.
    - documents: List of all documents in the corpus.
    - avg_doc_length: The average document length in index terms.
    - output_file_name: Name of the file to save results (default is 'results.txt').
    - k1: BM25+ hyperparameter (default is 1.2)
    - b: BM25+ hyperparameter (default is 0.75)
    - delta: BM25+ hyperparameter (default is 1)
    - top_n: Maximum number of top documents to retrieve for each query (default is 100).
    - run_tag: A unique identifier for this run.
    """
    
    with open(output_file_name, "w") as output_file:
        for query in queries:
            query = Query(_id=query['_id'], query=query['text'])

            # Perform a ranking again of the documents
            top_documents = bm25_rank_documents_for_query(query, inv_index, document_vectors, documents, avg_doc_length, k1=k1, b=b, delta=delta, top_n=top_n)

            # Write results in the required format
            for rank, (doc_id, score) in enumerate(top_documents, start=1):
                output_file.write(f"{query.get_id()} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")

            print(f"Top results for Query {query.get_id()}:")
            for rank, (doc_id, score) in enumerate(top_documents[:5], start=1):  # Display top 5 for debugging
                print(f"Rank {rank}: Document ID {doc_id}, Score {score:.6f}")
            print("")

    print(f"Results have been saved to {output_file_name}.")

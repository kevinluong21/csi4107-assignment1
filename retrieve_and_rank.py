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

    # Initialize a dictionary to store similarity scores
    similarities = {}

    for doc_id, doc_vector in document_vectors.items():
        # Check if the document contains at least one of the query words using the inverted index
        contains_query_word = False
        for word in query.split():  
            if word in inverted_index.index:  # Access the index of the inverted index
                contains_query_word = True
                break

        # If the document contains query words, compute similarity
        if contains_query_word:
            similarity = compute_cosine_similarity(query_vector, doc_vector)
            if similarity > 0:  # Only consider documents with a non-zero similarity
                similarities[doc_id] = similarity

    # Sort the documents by similarity score in descending order
    sorted_documents = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    if not sorted_documents:
        print(f"No documents returned for query: {query}")
    top_documents = sorted_documents[:top_n]

    return top_documents

def pseudo_relevance_loop(query, documents:dict[int, Document], top_documents:list, n=2, k=2):
    """
    Take the top n terms of the top k documents returned by the first pass of the IR and add them to the end of the query.

    Parameters:
        query (str): The query as a string.
        documents (dict): A dictionary where the document ID is the key and the Document object is the value.
        top_documents (list): A list of top documents where each index is a tuple of the document ID and the similarity score.
        n: The top number of words to extract from each of the top k documents. By default, 2.
        k: The top k documents to extract terms from. By default, 2.

    Returns:
        query (str): The query with more terms appended to it.
    """
    # Split query into words and make it into a set to avoid duplicating terms to add and terms already in the query
    query = query.split(" ")
    query = set(query)

    terms_to_add = set()

    for document in top_documents[:k]:
        _id = document[0]
        index_terms = documents[_id].get_index_terms()
        # Sort the index terms by count
        index_terms = sorted(index_terms.items(), key=lambda item: item[1], reverse=True)
        
        # To avoid adding duplicate terms into the query but still adding n terms to the query, continue looping through the index
        # terms until you find a term that is not already in the query
        i = 0
        j = 0
        while i < n and j < len(index_terms):
            term = index_terms[j][0]
            if term not in query:
                terms_to_add.add(term)
                i += 1
                j += 1
            else:
                j += 1

    query = " ".join(query) + " " + " ".join(terms_to_add)
    query = query.strip()
    return query
  
def process_and_save_results(queries, inv_index, document_vectors, documents, output_file_name="Results", top_n=100, run_tag="run1"):
    """
    Process queries, rank documents, and save the top results in the required format.

    Output Format:
    query_id Q0 doc_id rank score run_tag

    Parameters:
    - queries: List of query dictionaries containing '_id' and 'text'.
    - inv_index: Inverted index used for retrieving relevant documents.
    - document_vectors: Precomputed document vectors for similarity calculation.
    - documents: List of all documents in the corpus.
    - output_file_name: Name of the file to save results (default is 'Results').
    - top_n: Maximum number of top documents to retrieve for each query (default is 100).
    - run_tag: A unique identifier for this run.
    """
    with open(output_file_name, "w") as output_file:
        for query in queries:
            query_id = query['_id']  # Query ID
            query_text = query['text']  # Query text
            
            # print(f"Processing query {query_id}: {query_text}")
            
            # Rank documents for the query
            top_documents = rank_documents_for_query(query_text, inv_index, document_vectors, len(documents), top_n=top_n)

            # Perform a pseudo-relevance feedback loop
            print(query_text)
            query_text = pseudo_relevance_loop(query_text, documents, top_documents)
            print(query_text)
            top_documents = rank_documents_for_query(query_text, inv_index, document_vectors, len(documents), top_n=top_n)
            
            # Write results in the required format
            for rank, (doc_id, score) in enumerate(top_documents, start=1):
                output_file.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")

            print(f"Top results for Query {query_id}:")
            for rank, (doc_id, score) in enumerate(top_documents[:5], start=1):  # Display top 5 for debugging
                print(f"Rank {rank}: Document ID {doc_id}, Score {score:.6f}")
            print("")

    print(f"Results have been saved to {output_file_name}.")

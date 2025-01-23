from preprocessing import Document
from collections import defaultdict

class InvertedIndex:

    def __init__(self):
        self.index = defaultdict(lambda: defaultdict(int)) #term -> doc_id -> frequency
    
    def add_documents(self, doc_id: int, terms: dict):
        ''' Add document's terms to the inverted index.
        Parameters:
        doc_id (int): ID of the document
        terms (dict): the dictionary of terms with their frequencies'''

        for term, freq in terms.items():
            self.index[term][doc_id] += freq

    def get_postings(self, term: str):
        '''Get postings list for a term.
        Posting refers to an entry in the index that indicates the documents where a term appears, along with term frequency

        term (str): term obtained from tokenization step 

        Returns:
            dict: a dictionary of document IDs and their term frequencies
        '''

        return self.index.get(term, {})

    def __repr__(self):
        return "\n".join(f"{term}: {dict(postings)}" for term, postings in self.index.items())
    

# doc1 = Document(title="AI and Machine Learning", text="AI and ML are closely related fields.")
# doc2 = Document(title="Deep Learning", text="Deep Learning is a subset of Machine Learning.")
# doc3 = Document(title="Artificial Intelligence", text="AI covers a wide range of topics.")


# index_terms_doc1 = doc1.get_index_terms()
# index_terms_doc2 = doc2.get_index_terms()
# index_terms_doc3 = doc3.get_index_terms()


# inv_index = InvertedIndex()


# inv_index.add_documents(1, index_terms_doc1)
# inv_index.add_documents(2, index_terms_doc2)
# inv_index.add_documents(3, index_terms_doc3)


# print("Postings for 'AI':", inv_index.get_postings('ai'))
# print("Postings for 'Machine':", inv_index.get_postings('machine'))


# print("\nFull Inverted Index:")
# print(inv_index)

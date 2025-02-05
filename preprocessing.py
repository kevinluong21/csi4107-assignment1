import pandas as pd
import nltk
import string
import re
from collections import Counter, defaultdict
from spellchecker import SpellChecker
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# TODO: add every synonym of each index term with the same frequency using wordnet!

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

# Used to split text into words without separating the punctuation (terms with hyphens remain intact because of words like "pre-diabetes" and "body-mass"). The punctuation is also maintained for each word in the case there is no space between sentences which results in cases like "I like to read.I like hats." where "read" and "I" should be 2 separate words.
word_splitter = RegexpTokenizer(pattern=r"\w+[-]\w+|\w+['.,!?]*|\w+|\S+")

# To check if a string is valid word (when splitting hyphenated words), use a spellchecker
spell = SpellChecker()

# Used to lemmatize (a form of stemming) words (better performance than the Porter and Lancaster stemmer)
lemmatizer = WordNetLemmatizer()

def is_hyphenated_compound_word(word:str) -> bool:
    '''
    Returns True if the strings in a hyphenated word, when split by its hyphens, are all actual words (a compound word). Otherwise, returns False.

    e.g. "pre-diabetes" is False, "hello" is False (does not contain a hyphen), and "body-mass" is True

    Parameters:
        word (str): Hyphenated word to test
    Returns:
        True or False
    '''
    # For hyphenated words, check if they are the combination of multiple "real" words together or just a prefix and/or suffix. If they are a combination of "real" words, split them into multiple words. Otherwise, remove the hyphen and make them into a single word.
    if "-" in word:
        terms = word.split("-")
        check_terms = [True if term in spell else False for term in terms]
        if all(check_terms):
            return True
    return False

# def is_number(s:str) -> bool:
#     '''
#     Returns True if a string is a number or is a string of numbers with decimal points (e.g. 123.456.789). Otherwise, returns False.

#     Parameters:
#         s (str): String to test
#     Returns:
#         True or False
#     '''
#     # Remove all the decimal points and check if the string contains only numbers
#     try:
#         s = s.replace(".", "")
#         int(s)
#         return True
#     except:
#         return False
    
def generate_synonyms(word:str) -> list[str]:
    synsets = wordnet.synsets(word)

    if not synsets:
        return []

    synonyms = {word}

    for synset in synsets:
        for lemma in synset.lemmas():
            synonym = lemma.name().lower().replace("_", " ")
            if is_hyphenated_compound_word(synonym):
                synonym = set(re.split(pattern=r"[\s-]", string=synonym))
            else:
                synonym = set(synonym.split(" "))

            synonyms = synonyms.union(synonym)

    synonyms.discard(word)

    synonyms = synonyms.difference(stop_words)
    synonyms = list(synonyms)
    # Remove any non-letters (except for hyphens)
    synonyms = [re.sub(pattern=r'[^\x61-\x7A-]', string=word, repl="") for word in synonyms]
    # Remove any empty strings
    synonyms = [word for word in synonyms if word]
    # Lemmatize each synonym
    synonyms = [lemmatizer.lemmatize(word) for word in synonyms]
    # # Remove any numbers
    # synonyms = [word for word in synonyms if not is_number(word)]

    return synonyms

def extract_index_terms(text:str) -> dict[str: int]:
    '''
    Given a string, extracts all of the relevant index terms along with their term frequencies, ignoring numbers, punctuation, and stopwords.

    Parameters:
        text (str): String to extract index terms
    Returns:
        index_terms (dict): A dictionary containing index terms as keys and its term frequency within the document as values.
    '''

    if not text:
        return {}
    
    text = text.lower().strip()
    # If there are any unicode characters in the text, decode them into their proper representations
    text = text.encode('unicode_escape').decode('unicode_escape')
    # Splits the given text into words
    words = word_splitter.tokenize(text)
    # Remove all non-letters from the string entirely (maintain hyphens)
    words = [re.sub(pattern=r'[^\x61-\x7A-]', string=word, repl="") for word in words]
    # If a hyphenated word is a compound word, split the word. If not, remove the hyphen.
    words = [word.split("-") if is_hyphenated_compound_word(word) else [word.replace("-", "")] for word in words]
    words = [word for list_of_words in words for word in list_of_words] # Flatten the list of lists into a single list
    # Remove any empty strings
    words = [word for word in words if word]
    # # Remove any numbers
    # words = [word for word in words if not is_number(word)]

    # Count the occurences of each word within the document
    term_freq = Counter(words)
    index_terms = dict()

    # Using set difference, remove all the stopwords
    words = set(term_freq.keys())
    words = words.difference(stop_words)

    # For each word, lemmatize and then combine the counts for words that, after lemmatization, match with an existing word in the index
    for key in term_freq.keys():
        if key not in words:
            continue

        root_word = lemmatizer.lemmatize(key)
        if root_word == key:
            if key in index_terms.keys():
                index_terms[key] += term_freq[key]
            else:
                index_terms[key] = term_freq[key]
        else:
            if root_word in index_terms.keys():
                index_terms[root_word] += term_freq[key]
            else:
                index_terms[root_word] = term_freq[key]

    return index_terms


class RetrievalItem:
    _id = -1

    def __init__(self, text, _id=None):        
        if _id is None:
            self._id = Document.increment_id()
        else:
            self._id = _id #use the id passed to the doc

        self.index_terms = extract_index_terms(text)

    @classmethod
    def increment_id(cls):
        '''
        If no ID is provided, increment the ID class variable and set as the default ID for the document.

        Parameters:
            None
        Returns:
            _id (int): The incremented ID
        '''
        cls._id += 1
        return cls._id
    
    def get_id(self):
        '''
        Returns the ID of the Document.

        Parameters:
            None
        Returns:
            _id (int): The ID of the Document.
        '''
        return self._id
    
    def get_index_terms(self):
        '''
        Returns the index terms and term frequency of the document.

        Parameters:
            None
        Returns:
            index_terms (dict): A dictionary containing index terms as keys and its term frequency within the document as values.
        '''
        return self.index_terms
    
    def __len__(self):
        """
        Returns the length of an object in index terms.
        """
        return len(self.index_terms)

class Document(RetrievalItem):
    _id = -1

    def __init__(self, title, text, _id=None, metadata={}):
        self.title = title.strip()
        self.text = text.strip()
        
        super().__init__(self.title + " " + self.text, _id)

        self.metadata = metadata

    def get_title(self):
        return self.title
    
    def get_text(self):
        return self.text

    def __repr__(self):
        return f"Document(id={self._id}, title={self.title}, text={self.text}, index={self.index_terms}, metadata={self.metadata})"

class Query(RetrievalItem):

    def __init__(self, query, _id=None):
        self.query = query.strip()

        super().__init__(self.query, _id)

        self.index_synonyms = {}

        for term in self.index_terms:
            self.index_synonyms[term] = generate_synonyms(term)

    def get_query(self):
        return self.query
    
    def get_index_synonyms(self):
        """
        Returns the synonyms for every index term using WordNet.

        Parameters:
            None
        Returns:
            index_synonyms (dict): A dictionary containing index terms as keys and all of its synonyms as values.
        """
        return self.index_synonyms

    def __repr__(self):
        return f"Query(id={self._id}, query={self.query}, index={self.index_terms}, synonyms={self.index_synonyms})"

# document = Document(_id= "4983", title= "Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging.", text= "Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical development and result in functional disabilities. A line scan diffusion-weighted magnetic resonance imaging (MRI) sequence with diffusion tensor analysis was applied to measure the apparent diffusion coefficient, to calculate relative anisotropy, and to delineate three-dimensional fiber architecture in cerebral white matter in preterm (n = 17) and full-term infants (n = 7). To assess effects of prematurity on cerebral white matter development, early gestation preterm infants (n = 10) were studied a second time at term. In the central white matter the mean apparent diffusion coefficient at 28 wk was high, 1.8 microm2/ms, and decreased toward term to 1.2 microm2/ms. In the posterior limb of the internal capsule, the mean apparent diffusion coefficients at both times were similar (1.2 versus 1.1 microm2/ms). Relative anisotropy was higher the closer birth was to term with greater absolute values in the internal capsule than in the central white matter. Preterm infants at term showed higher mean diffusion coefficients in the central white matter (1.4 +/- 0.24 versus 1.15 +/- 0.09 microm2/ms, p = 0.016) and lower relative anisotropy in both areas compared with full-term infants (white matter, 10.9 +/- 0.6 versus 22.9 +/- 3.0%, p = 0.001; internal capsule, 24.0 +/- 4.44 versus 33.1 +/- 0.6% p = 0.006). Nonmyelinated fibers in the corpus callosum were visible by diffusion tensor MRI as early as 28 wk; full-term and preterm infants at term showed marked differences in white matter fiber organization. The data indicate that quantitative assessment of water diffusion by diffusion tensor MRI provides insight into microstructural development in cerebral white matter in living infants.", metadata= {})

# print(len(document))
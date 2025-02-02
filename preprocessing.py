import pandas as pd
import nltk
import string
import re
from collections import Counter, defaultdict
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

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

def is_number(s:str) -> bool:
    '''
    Returns True if a string is a number or is a string of numbers with decimal points (e.g. 123.456.789). Otherwise, returns False.

    Parameters:
        s (str): String to test
    Returns:
        True or False
    '''
    # Remove all the decimal points and check if the string contains only numbers
    try:
        s = s.replace(".", "")
        int(s)
        return True
    except:
        return False

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
    # Remove all non-letters from the string entirely
    words = [re.sub(pattern=r'[^\x61-\x7A-]', string=word, repl="") for word in words]
    # Remove any empty strings
    words = [word for word in words if word]
    # If a hyphenated word is a compound word, split the word. If not, remove the hyphen.
    words = [word.split("-") if is_hyphenated_compound_word(word) else [word.replace("-", "")] for word in words]
    words = [word for list_of_words in words for word in list_of_words] # Flatten the list of lists into a single list
    # Remove any numbers
    words = [word for word in words if not is_number(word)]
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

class Document:
    _id = -1

    def __init__(self, title, text, _id=None, metadata={}):
        self.title = title
        self.text = text
        
        if _id is None:
            self._id = Document.increment_id()
        else:
            self._id = _id #use the id passed to the doc

        self.metadata = metadata
        self.index_terms = extract_index_terms(self.title.strip() + " " + self.text.strip())

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
    
    def get_index_terms(self):
        '''
        Returns the index terms and term frequency of the document.

        Parameters:
            None
        Returns:
            index_terms (dict): A dictionary containing index terms as keys and its term frequency within the document as values.
        '''
        return self.index_terms

    def __repr__(self):
        return f"Document(id={self._id}, title={self.title}, text={self.text}, index={self.index_terms}, metadata={self.metadata})"

# Assignment 1 (CSI-4107)
## Members of Group 25
| Member        | Student Number |
| ------------- | -------------- |
| Nalan Kurnaz  | 300245521      |
| Alona Petrova | 300074852      |
| Kevin Luong   | 300232125      |

## Contributions
| Member        | Contributions                                                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Nalan Kurnaz  |                                                                                                                                                              |
| Alona Petrova |                                                                                                                                                              |
| Kevin Luong   | - Worked on the pre-processing<br>- Implemented BM25+<br>- Investigate synonym replacement and pseudo-relevance feedback feasibility<br>- Work on report<br> |

## Functionality
Our Information Retrieval system takes as input documents and a query and returns and ranks the top 100 documents returned for a query using a BM25+ algorithm. 

## Get Started
## Analysis of Algorithms, Data Structures, and Optimizations

### Algorithms
#### Pre-processing
To extract index terms and count term frequencies, the algorithm follows:
- Lower and strip the original text.
- To get rid of any unicode characters, we need to first encode and decode the characters in order to properly remove them. Otherwise, they are stored in the string as actual unicodes such as "\u2013".
- Split the text using a regex word splitter that splits on spaces and punctuation, but maintains hyphens. The punctuation will be removed later.
- Remove all non-letter characters (except for the hyphen).
- For terms that have a hyphen, determine if it is a compound word connected by a hyphen or a word composed of an affix and a root word. If it is the former, split the word into 2 terms. If not, remove the hyphen and keep the word as 1 term.
- Count the term frequencies of each term and return them as a dictionary with the term as key and the term frequency as the value.
- Remove all stopwords from index terms by using a set difference.
- Lemmatize each term. If, after lemmatization, the term happens to match an existing term (e.g. "flies" lemmatizes to "fly"), then combine their term frequencies.
- Return the dictionary of index terms and term frequencies.

#### Ranking
To calculate the BM25+ weighting of each term, the BM25 formula by Robertson and Sparck-Jones (1976) was modified by adding a new parameter, delta:
$$
\text{weight} = \frac{( \text{term\_freq} + \delta ) \times \log\left(\frac{\text{total\_documents} - \text{doc\_freq} + 0.5}{\text{doc\_freq} + 0.5} \right)}
{(k_1 \times ((1 - b) + (b \times \text{doc\_length} / \text{avg\_doc\_length}))) + \text{term\_freq}}
$$

### Data Structures
#### Pre-processing
For the documents and queries, we created a parent class called RetrievalItem and had a class Document and Query inherit that class. By doing this, we only needed to define variables like IDs and text/queries once and functions like extracting index terms once. At initialization of a new document of query, all of the information pertaining to its ID, text content, and index terms with its term frequency are all stored in the object, so we do not need to compute the index terms and term frequency each time. Furthermore, this will facilitate creating and indexing an inverted index because the terms and term frequencies will always be stored within the object.

To store the index terms within each Document and Query object, the index terms are stored as a dictionary in which the key is the index term and the value is the term frequency for the document/query.

### Optimizations
#### Pre-processing
- When splitting words, we used the NLTK Regex Word Splitter instead of opting simply for Python's string split function. This is to avoid cases where terms may be stuck together because of a missing space after an ending punctuation mark (e.g. "class.The" should be 2 terms). Therefore, we decided to use regex to split based on punctuation marks and spaces, but maintain hyphens because of compound words.
- While exploring the documents, we found that certain terms that were hyphenated may actually be a compound word whereas some were simply words with prefixes. In the case of terms like "body-mass" which is a compound word, we found that it would sometimes appear as "body mass" as well. To ensure an accurate term frequency and document frequency, we created a check that would split hyphenated words into parts and then check for each part to see if it was an actual word on its own using the Spellchecker library. If each part is a word, we would split them into their own words. Otherwise, we kept them as a single term and removed the hyphen.
- For word stemming, we opted for lemmatization because it left room for future integration with synonym replacement. Since stemming would simply cut the end off of a word, the result could be a non-word (e.g. "flies" becomes "flie" which is not a word). By using lemmatization, we were able to output a word that could be fed into a domain-specific thesaurus like MeSH. Unfortunately, we could not gain access to a dictionary API without guaranteeing that 1) the API would always work and 2) that the API key would still be valid. Therefore, we left room for, if that were to happen, we could still use a thesaurus for synonym replacement. On the other hand, using WordNet, though local, does not contain enough domain-specific synonyms to best match a word in the inverted index and the word in the query.

#### Ranking
- When calculating the weight for each term, we opted for BM25, specifically the BM25+ variant, instead of TF-IDF. This is because BM25 considers factors like document length and can be fine-tuned using its hyperparameters k1 and b. We found that document length may vary which, to improve performance, needs to be considered in our ranking algorithm. We then chose the BM25+ variant that address the issue with BM25 where long documents that match a query term may be ranked the same as shorter documents that do not contain a query term at all. Therefore, we introduced the delta hyperparameter to control this behaviour. As a result of our testing, we have found that it has improved our MAP.
- We explored the possibility of using synonym replacement using WordNet, but found that MAP was decreasing. We found that certain queries did not match with the correct document because the query would contain terms that did not exactly match any term in the document. To circumvent this issue, we wanted to see if we could try substituting query terms that had 0 term frequency within the document with a synonym that appeared at least once in the document. The issue here is that 1) a lot of the query terms are domain-specific, so WordNet is unable to find the exact synonym and 2) oftentimes, generic terms were overfit to the document and therefore artificially inflated its similarity score. Therefore, this had caused a decrease in our MAP.
- We also explored the possibility of using a pseudo-relevance feedback loop. However, we found that our MAP was also decreasing. This follows the principle of "trash-in-trash-out" because if our top results were not relevant enough, then performing a feedback loop of the top n documents would simply fit our query to return more irrelevant documents. This resulted in a drop in our MAP.

## Results
### Vocabulary
### Output
### Evaluation: Mean Average Precision (MAP)
| Document Content | MAP (BM25+) |
| ---------------- | ----------- |
| Titles           |             |
| Titles + Text    | 0.5634      |

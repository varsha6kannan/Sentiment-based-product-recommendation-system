import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def preprocess(document, stem=True):
    # lower case all the words
    document = document.lower()
    
    # remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    document = document.translate(translator)
    
    # tokenize the words
    words = word_tokenize(document)
    
    # remove all the stop words
    words = [word for word in words if word not in stopwords.words("english")]
    
    # bring the word to the root form
    if stem:
        words = [stemmer.stem(word) for word in words]
    else:
        words = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in words]
    
    document = " ".join(words)
    return document

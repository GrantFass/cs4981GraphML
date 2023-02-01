import numpy as np
import contractions
import unidecode
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
from collections import Counter
#KDE-LDA paper? Incorporating Knowledge Graph Embeddings into Topic Modeling
#Variational Graph AutoEncoding

class CorpusBuilder:
    
    def add_spaces_around_parens(self, text: str, remove_actions=True):
        """Method to add spaces around parentheses. This will add the spaces to the outside of the parentheses and not affect the content inside the parentheses.
        This is primarily used to clean up common phrases in the TEDTalksDataset. This method also gives an option to remove some of the common phrases entirely.

        Args:
            text (str): the text to add spaces around the parentheses to.
            remove_actions (bool, optional): gives the option to remove most of the commonly occuring action words inside parentheses. Mostly used for the TEDTalksDataset. Defaults to True.

        Returns:
            str: the cleaned up text.
        """
        remove_list = ['(Laughter)', '(laughter)', '(Music)', '(music)', '(Music ends)', '(Audience cheers)', '(Applause)', '(Applause ends)', '(Applause continues)', '(Bells)', '(Trumpet)', '(Clears throat)']
        for i in remove_list:
            text = text.replace(i, ' ')
        text = text.replace("(", " (")
        text = text.replace(")", ") ")
        return text.replace("  ", ' ')

    def fix_punctuation_spacing(self, text: str):
        """Method to fix the spacing around punctuations. This is mostly used to add spaces after periods and commas.
        The regex in this method was sourced from [a stackoverflow post](https://stackoverflow.com/questions/44263446/python-regex-to-add-space-after-dot-or-comma)

        Args:
            text (str): the text to update the spacing around punctuations.

        Returns:
            str: the cleaned up text.
        """
        text = re.sub(r'(?<=[\.\,\?])(?=[^\s])', r' ', text)
        return text
    
    def strip_punctuation(self, text: str):
        """Method to strip the punctuation from a string
        
        Args:
            text (str): the text to strip punctuation from
            
        Returns:
            str: the cleaned up text
        """
        return re.sub(r'[\.\,\?\\\/\<\>\;\:\[\]\{\}]', r'', text)
    
    # def __init__(self, data: list(str)):
    def clean(self, data: list(str)):
        # builds unigram corpus
        # expects a pandas series of strings to be passed
        
        # remove any html tags
        data = list(map(lambda x: BeautifulSoup(x, "html.parser").get_text(separator=" ")), data)
        # clean up the contractions
        data = list(map(lambda x: contractions.fix(x), data))
        # remove accended characters
        data = list(map(lambda x: unidecode.unidecode(x), data))
        # remove stopwords: https://stackoverflow.com/questions/19560498/faster-way-to-remove-stop-words-in-python
        cachedStopWords = Counter(stopwords.words('english'))
        data = list(map(lambda x: ' '.join([word for word in x.split() if word not in cachedStopWords])), data)
        # remove extra whitespace
        data = list(map(lambda x: ' '.join(x.strip().split()), data))
        
        
            
        
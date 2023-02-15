"""
## Cleaning and Preprocessing Section
There are a few common preprocessing steps that can be performed. Some were found from [this TowardsDataScience article](https://towardsdatascience.com/nlp-text-preprocessing-a-practical-guide-and-template-d80874676e79).

0. Extra Preprocessing Depending on Source
    - Remove HTML tags
    - Convert accented characters to ASCII characters
1. Contraction Expansion
    - expand contractions so that they are two words instead of one.
2. Convert Word Form of Numbers to Numerics
    - i.e., convert four thousand to 4000
3. Optionally may want to remove numbers altogether.
4. Remove Special Characters
5. Tokenization
    - This involves splitting the text into sentences and then splitting the sentences into words.
    - `from nltk.tokenize import word_tokenize`
    - `words = word_tokenize(sentence)`
6. Punctuation Cleaning
    - This involves cleaning all of the punctuation out of the words.
7. Short Word Removal
    - This involves removing all of the words that are shorter than 3 characters as they are commonly not useful.
    - Note that this is done before the case cleaning so that a check can be performed if all letters are capitalized which would signify an acronym.
8. Case Cleaning
    - This involves making all of the words the same case. Usually lower case.
    - `sentence = sentence.lower()`
9. Stopword Removal
    - This involves removing all of the stopwords from the text. Stop Words are a set of commonly used words in language. These words are so common that they carry very little useful information.
    - Note that some stopwords are actually important to the meaning of the text and may want to be added back in. I.E., the word 'not' in sentiment analysis.
    - `from nltk.corpus import stopwords`
    - `stop_words = set(stopwords.words('english'))`
    - `filtered_sentence = [w for w in word_tokens if not w in stop_words]`
10. Lemmatization
    - change all of the third person words to first person
    - change all verbs in past and future tenses to present tense.
    - `fron nltk.stem import WordNetLemmatizer`
    - `lemmatizer = WordNetLemmatizer()`
    - `lemmatizer.lemmatize('Machine", pos='n')`
11. Stemming
    - reduce words to their root form.
    - May not always be wanted as it can reduce words to far so that they are no longer words and no longer make sense. I.E., 'machine' -> 'machin'
    - `from nltk.stem import PorterStemmer`
    - `ps = PorterStemmer()`
    - `ps.stem(word)`
12. Remove Extra Whitespace
"""
# Load in the possible model sizes from [SpaCy](https://spacy.io/models/en). Download using the command `python -m spacy download en_core_web_sm`
import en_core_web_sm # 12.8 Mb
import en_core_web_md # 42.8 Mb
import en_core_web_lg # 587.7 Mb
# import en_core_web_trf # 460.3 Mb This one is not needed as it is more of a transformer pipeline than a preprocessing and cleaning pipeline.

# These are the other SpaCy imports that are used for cleaning.
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.language import Language

# These are the non SpaCy imports that are used for cleaning.
import unidecode
import contractions
from word2number import w2n
from bs4 import BeautifulSoup
import re
from collections import Counter
from nltk.corpus import stopwords

class Preprocessor:
    # define nlp as a variable so the preprocessor size list can be specified dynamically based on file.
    nlp = 0
    
    def __init__(self, size: int, trigrams=False):
        """This is the constructor for the preprocessor class. This is mainly used for defining the size of the preprocessor to use as stored in the `nlp` variable.
        Furthermore, this method makes sure that the named entity recognition pipeline step and the merge noun chunks step are both added to the pipeline.
        The merge noun chunks step comes from [a stackoverflow post about bigrams and trigrams consolidation](https://stackoverflow.com/questions/53598243/is-there-a-bi-gram-or-tri-gram-feature-in-spacy).
        Args:
            size (int): This variable is used to define the size of the nlp wordlist to use. The default is the smallest set. 1 defines the medium set. 2 defines the large set.
        """
        if size == 1:
            self.nlp = spacy.load('en_core_web_md')
        elif size == 2:
            self.nlp = spacy.load('en_core_web_lg')
        else:
            self.nlp = spacy.load('en_core_web_sm')
            
        config = {
        "moves": None,
        "update_with_oracle_cut_size": 100,
        "model": DEFAULT_NER_MODEL,
        "incorrect_spans_key": "incorrect_spans",
        }
        new_name = 'ner'
        if not new_name in self.nlp.pipe_names:
            self.nlp.add_pipe(new_name, config=config)
        if trigrams:
            new_name = 'merge_noun_chunks'
            if not new_name in self.nlp.pipe_names:
                self.nlp.add_pipe(new_name)
        
    def remove_stop_words(self, stop_words: list[str]):
        """This method is used to remove certain words from the list of stop words present in the SpaCy list of stop words to remove from the text.

        Args:
            stop_words (list[str]): The list of stop words (formatted as strings) to remove from the SpaCy `nlp` vocab list.
        """
        for stop_word in stop_words:
            self.nlp.vocab[stop_word].is_stop = False
            
    def add_stop_words(self, stop_words: list[str]):
        """This method is used to add certain words to the SpaCy stored stop words vocab list stored under the `nlp` variable.

        Args:
            stop_words (list[str]): The list of stop words (formatted as strings) to add to the SpaCy `nlp` vocab list.
        """
        for stop_word in stop_words:
            self.nlp.vocab[stop_word].is_stop = True

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
    
    def clean_base(self, x: str):
        # remove any html tags
        x = BeautifulSoup(x, "html.parser").get_text(separator=" ")
        # set all to lower
        x = x.lower()
        # clean up the contractions
        x = contractions.fix(x)
        # remove accended characters
        x = unidecode.unidecode(x)
        # remove stopwords: https://stackoverflow.com/questions/19560498/faster-way-to-remove-stop-words-in-python
        cachedStopWords = Counter(stopwords.words('english'))
        x = ' '.join([word for word in x.split() if word not in cachedStopWords]) # slower to use word tokenize
        # # fix punctuation spacing
        # x = re.sub(r'(?<=[\.\,\?])(?=[^\s])', r' ', x)
        # # strip punctuation
        # x = re.sub(r'[\.\,\?\\\/\<\>\;\:\[\]\{\}]', r'', x)
        # strip quotes
        x = x.replace('\'', '').replace('\"', '')
        # remove some actions
        remove_list = ['\\r\\n', '\r\n', '[Instructor]', '[Voiceover]', '[instructor]', '[voiceover]', '(Laughter)', '(laughter)', '(Music)', '(music)', '(Music ends)', '(Audience cheers)', '(Applause)', '(Applause ends)', '(Applause continues)', '(Bells)', '(Trumpet)', '(Clears throat)']
        for word in remove_list:
            x = x.replace(word, ' ')
        # x = ' '.join([word for word in x.split() if word not in remove_list])
        # remove extraneous items
        x = x.replace(' -- ', '').replace(' .. ', ' ').replace(' ... ', ' ')
        # remove extra whitespace
        x = ' '.join(x.strip().split())
        return x
    
    def clean(self, text: str, tokenize=False, verbose_tokenize=False, fast=False):
        text = self.clean_base(text)
        if not tokenize and not fast:
            doc = self.nlp(text)
            return doc
        elif verbose_tokenize and tokenize:
            doc = self.nlp(text)
            clean_text = []
            for token in doc:
                flag = True
                edit = token.text
                # remove stop words
                if token.is_stop and token.pos_ != 'NUM': 
                    flag = False
                # remove punctuations
                if token.pos_ == 'PUNCT' and flag == True: 
                    flag = False
                # remove special characters
                if token.pos_ == 'SYM' and flag == True: 
                    flag = False
                # remove numbers
                if (token.pos_ == 'NUM' or token.text.isnumeric()) \
                and flag == True:
                    flag = False
                # convert number words to numeric numbers
                if token.pos_ == 'NUM' and flag == True:
                    edit = w2n.word_to_num(token.text)
                    token.text = edit # may cause issues. Unsure
                # convert tokens to base form
                elif token.lemma_ != "-PRON-" and flag == True:
                    edit = token.lemma_
                # append tokens edited and not removed to list 
                if edit != "" and flag == True and len(token.text) > 3:
                    clean_text.append(edit)    
            return clean_text
        elif not verbose_tokenize and tokenize:
            return text.split()
        else:
            return text
    
    # def clean(self, text: str, accented_chars=True, contractions=True, 
    #                    convert_num=True, extra_whitespace=True, 
    #                    lemmatization=True, remove_short_words=True, lowercase=True, punctuations=True,
    #                    remove_html=True, remove_num=True, special_chars=True, 
    #                    stop_words=True, get_doc=False):
    #     """Method to clean up and preprocess text to a standard format.

    #     Args:
    #         text (str): the text to clean up
    #         accented_chars (bool, optional): defines if accented characters are removed / standardized. Defaults to True.
    #         contractions (bool, optional): defines if contractions are expanded. Defaults to True.
    #         convert_num (bool, optional): defines if numbers are standardized to their word form. Defaults to True.
    #         extra_whitespace (bool, optional): defines if extra whitespace is removed. Defaults to True.
    #         lemmatization (bool, optional): defines if the text is lemamtized. Defaults to True.
    #         remove_short_words (bool, optional): defines if words under a certain length are removed. Defaults to True.
    #         lowercase (bool, optional): defines if text is defaulted to lowercase. Defaults to True.
    #         punctuations (bool, optional): defines if punctuation is removed. Defaults to True.
    #         remove_html (bool, optional): defines if HTML tags are removed. Defaults to True.
    #         remove_num (bool, optional): defines if numbers should be removed. Defaults to True.
    #         special_chars (bool, optional): defines if special characters should be removed. Defaults to True.
    #         stop_words (bool, optional): defines if stop words should be removed. Defaults to True.
    #         get_doc (bool, optional): defines if the method should return the SpaCy document or return the cleaned text directly. Defaults to False.
            
    #     Returns:
    #         Returned value depends on the value of the get_doc method. 
    #         If the value is True the function will return the SpaCy doc directly. 
    #         Othewise the function will return the cleaned up text.
    #         str: the cleaned up text.
    #         spacy.tokens.doc.Doc: the SpaCy document file.
    #     """
    #     text = self.add_spaces_around_parens(text)
    #     text = self.fix_punctuation_spacing(text)
    #     if remove_html == True: #remove html tags
    #         text = self.strip_html_tags(text)
    #     if extra_whitespace == True: #remove extra whitespaces
    #         text = self.remove_whitespace(text)
    #     if accented_chars == True: #remove accented characters
    #         text = self.remove_accented_chars(text)
    #     if contractions == True: #expand contractions
    #         text = self.expand_contractions(text)
    #     if lowercase == True: #convert all characters to lowercase
    #         text = text.lower()
            
    #     if punctuations == True:
    #         text = self.strip_punctuation(text)
            

    #     doc = self.nlp(text) #tokenise text

    #     clean_text = []
        
    #     for token in doc:
    #         flag = True
    #         edit = token.text
    #         # remove stop words
    #         if stop_words == True and token.is_stop and token.pos_ != 'NUM': 
    #             flag = False
    #         # remove punctuations
    #         if punctuations == True and token.pos_ == 'PUNCT' and flag == True: 
    #             flag = False
    #         # remove special characters
    #         if special_chars == True and token.pos_ == 'SYM' and flag == True: 
    #             flag = False
    #         # remove numbers
    #         if remove_num == True and (token.pos_ == 'NUM' or token.text.isnumeric()) \
    #         and flag == True:
    #             flag = False
    #         # convert number words to numeric numbers
    #         if convert_num == True and token.pos_ == 'NUM' and flag == True:
    #             edit = w2n.word_to_num(token.text)
    #             token.text = edit # may cause issues. Unsure
    #         # convert tokens to base form
    #         elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
    #             edit = token.lemma_
    #         # append tokens edited and not removed to list 
    #         if edit != "" and flag == True and (remove_short_words == True and len(token.text) > 3):
    #             clean_text.append(edit)    
    #     if not get_doc:    
    #         return clean_text
    #     else:
    #         return doc
        
    def read_vtt_as_text(self, filepath: str):
        """Method to read in a VTT transcript file and convert it to a string.
        
        Args:
            filepath (str): Path to VTT transcript file
            
        Returns:
            The VTT transcript file stripped of all timestamps and authors as a string.
        """
        with open(filepath) as f:
            lines = f.readlines()
        lines = lines[2:] # remove the WebVTT header
        output_text = ""
        for i in range(0, len(lines), 3):
            ts = lines[i]
            # Timestamp caputure groups: r'(\d{1,2}\:\d{1,2}\:\d{1,2}\.\d{1,3})'
            txt = lines[i + 1]
            if re.match(r'<v\s\w*,\s\w*>[\w\s.,?!\'\"=+-_&]*<\/v>', txt):
                txt = re.sub(r'^<v\s\w*,\s\w*>', '', txt)
                txt = re.sub(r'<\/v>$', '', txt)
            output_text += txt
            # newline is i + 2
        return output_text
    
    def get_acronyms(self, text: str):
        """"Find all acronyms in the text and return them as a list (currently doesn't provide expansions, see issue)"""
        acs = list(set(re.findall(r"\b[A-Z\.]{3,}s?\b", text)))
        uniq_acs = []
        for ac in acs: #Could have the same acronym listed twice, one with periods, one without (unlikely, but should catch it)
            uniq_acs.append(ac.replace('.',''))
        return list(set(uniq_acs))


import re
import copy
from pymystem3 import Mystem
from gensim.matutils import corpus2csc
from gensim.corpora import Dictionary
from itertools import islice


class TextsTokenizer:
    """Tokenizer"""
    def __init__(self):
        self.m = Mystem()

    def texts2tokens(self, texts: [str]) -> [str]:
        """Lemmatization for texts in list. It returns list with lemmatized texts."""
        texts_ = [tx.replace("\n", " ") for tx in texts]
        text_ = "\n".join(texts_)
        text_ = re.sub(r"[^\w\n\s]", " ", text_)
        lm_texts = "".join(self.m.lemmatize(text_))
        return [lm_q.split() for lm_q in lm_texts.split("\n")][:-1]

    def __call__(self, texts: [str]):
        return self.texts2tokens(texts)


class TextShingles:
    """"""
    def __init__(self, size=3):
        self.size = size
        self.tokenizer = TextsTokenizer()

    def shingle_split(self, splited_texts: []) -> []:
        """"""
        shingles = []
        for splited_text in splited_texts:
            if len(splited_text) <= self.size:
                shingles.append(["".join(splited_text)])
            else:
                shingles_list = [list(islice(splited_text, int(i), int(i + self.size))) for i in range(len(splited_text))]
                shingles.append(["".join(l) for l in shingles_list if len(l) == self.size])
        return shingles

    def texts2shingles(self, texts:[str]):
        tokens = self.tokenizer(texts)
        return self.shingle_split(tokens)

    def __call__(self, texts):
        return self.texts2shingles(texts)


def tokens2vectors(tokens: [str], dictionary: Dictionary, max_dict_size: int):
    """"""
    corpus = [dictionary.doc2bow(lm_q) for lm_q in tokens]
    return [corpus2csc([x], num_terms=max_dict_size) for x in corpus]


class TokensVectors:
    """"""
    def __init__(self, max_dict_size: int):
        self.dictionary = None
        self.max_dict_size = max_dict_size

    def queries2vectors(self, tokens: []):
        """queries2vectors new_queries tuple: (text, query_id)
        return new vectors with query ids for sending in searcher"""
        # query_ids, texts = zip(*new_queries)

        if self.dictionary is None:
            gensim_dict_ = Dictionary(tokens)
            assert len(gensim_dict_) <= self.max_dict_size, "len(gensim_dict) must be less then max_dict_size"
            self.dictionary = Dictionary(tokens)
        else:
            gensim_dict_ = copy.deepcopy(self.dictionary)
            gensim_dict_.add_documents(tokens)
            if len(gensim_dict_) <= self.max_dict_size:
                self.dictionary = gensim_dict_

        return tokens2vectors(tokens, self.dictionary, self.max_dict_size)

    def __call__(self, new_queries):
        return self.queries2vectors(new_queries)


import pandas as pd
from ast import literal_eval

import gensim
from gensim.models import Phrases
from gensim.models.doc2vec import TaggedDocument
assert gensim.models.doc2vec.FAST_VERSION == 1, "this will be painfully slow otherwise"


def sentence_list(data):
    sentence_list = []
    brand_list = []
    for index, row in data.iterrows():
        review = [word for sentence in row['preprocessed'] for word in sentence]
        brand_list.append(row['brand'])
        sentence_list.append(review)
    return sentence_list, brand_list

def make_documents(sentence_list, brand_list, tagby, threshold = 10, min_count = 10):
    """
    tagby 가 true이면 tag에 sentence id
    false 이면 brand
    """
    bigram = Phrases(sentences=sentence_list, threshold=threshold, min_count=min_count)
    documents = []
    if tagby == True:
        for i in range(len(sentence_list)):
            document = TaggedDocument(words=bigram[sentence_list[i]], tags=[brand_list[i] + '_doc_'+str(i)])
            documents.append(document)
    if tagby == False:
        for i in range(len(sentence_list)):
            document = TaggedDocument(words=bigram[sentence_list[i]], tags=[brand_list[i]])
            documents.append(document)
    return documents, bigram

def load_data(save_path, category_list, tagby=True):
    # save_path = "E:/dataset/MasterThesis/FINAL/preprocess_data/"
    total = []
    for category in category_list:
        df = pd.read_csv(save_path + "preprocess_complete_" + category + ".csv")
        df['preprocessed'] = df.preprocessed.apply(lambda row: literal_eval(row))
        df_list, df_brands = sentence_list(df.sample(100000, random_state=42))
        documents, bigram = make_documents(df_list, df_brands, tagby=tagby)
        print("%s category is finished" % category)
        total += documents
    return total


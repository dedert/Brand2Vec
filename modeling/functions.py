from pandas import DataFrame
import numpy as np
import nltk
from collections import Counter
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_sim_words(model, brand, result_path, freq_dist, min_count, save=True, topn=20):
    df = DataFrame(columns=[['word', 'sim', 'freq']])
    result = model.most_similar([model.docvecs[brand]], topn=topn)
    if save:
        for tup in result:
            if freq_dist[tup[0]] >= min_count:
                df.loc[len(df)] = [tup[0], tup[1], freq_dist[tup[0]]]
        df.to_csv(result_path + 'keywords/' + brand + "_sim_words.csv", index=False)
        return
    else:
        for tup in result:
            if freq_dist[tup[0]] >= min_count:
                df.loc[len(df)] = [tup[0], tup[1], freq_dist[tup[0]]]
        return df


def extract_sim_brand(model, brand, result_path, save=True, topn=20):
    df = DataFrame(columns=[['word', 'sim']])
    result = model.docvecs.most_similar(brand, topn=topn)
    if save:
        for tup in result:
            df.loc[len(df)] = [tup[0], tup[1]]
        df.to_csv(result_path + 'keywords/' + brand + "_sim_brands.csv", index=False)
        return
    else:
        for tup in result:
            df.loc[len(df)] = [tup[0], tup[1]]
        return df

def cal_mean_cluster(df_result, cluster_idx, doc2vec_model, group_name='Cluster'):
    df = df_result[df_result[group_name] == cluster_idx]
    names = list(df['Name'].unique())
    all_arr = np.zeros((doc2vec_model.vector_size, len(names)))
    for index, name in enumerate(names):
        all_arr[:, index] = doc2vec_model.docvecs[name]
    return all_arr.mean(axis=1)


def print_result(vector, model, freq_dist, min_count, topn=50):
    df = DataFrame(columns=[['word','cos','freq']])
    lst = model.most_similar([vector], topn=topn)
    for tup in lst:
        if freq_dist[tup[0]] >= min_count:
            df.loc[len(df)] = [tup[0], tup[1], freq_dist[tup[0]]]
    return df

def save_brand_sim(model, sum_vector, name, save_path, topn=20):
    df = DataFrame(columns=('brand','sim'))
    lst = model.docvecs.most_similar([sum_vector], topn=topn)
    for tup in lst:
        df.loc[len(df)] = [tup[0], tup[1]]
    df.to_csv(save_path + name + '_simBrands.csv', index=False)
    return

# 각 브랜드의 단어 분포
def brand_raw_freq(documents, brand):
    brand_review = []
    for index, doc in enumerate(documents):
        if doc.tags[0] == brand:
            brand_review.append(doc.words)
    brand_review = [word for sent in brand_review for word in sent]
    corpus = nltk.Text(brand_review)
    freq = nltk.FreqDist(corpus)
    return brand_review, freq


def extract_keywords(score_df, brand, documents, selected, path, min_count = 100):
    keywords = score_df[['word',brand]].sort_values(brand, ascending=False)
    keywords.reset_index(inplace=True, drop=True)
    review, freq = brand_freq(documents, selected, brand)
    keyword_count = []
    df = DataFrame(columns=[["단어","확률유사도","빈도"]])
    for index, row in keywords.iterrows():
        if freq[row['word']] >= min_count:
            df.loc[len(df)] = [row['word'], row[brand], freq[row['word']]]
    df.to_csv(path + '/keywords/' + brand + '_Keywords.csv', index=False)


def brand_freq(documents, selected_words, brand):
    brand_review = []
    for index, doc in enumerate(documents):
        if doc.tags[0] == brand:
            brand_review.append(selected_words[index])
    brand_review = [word for sent in brand_review for word in sent]
    corpus = nltk.Text(brand_review)
    freq = nltk.FreqDist(corpus)
    return brand_review, freq


def clustering(model):
    brand_list = list(model.docvecs.doctags.keys())
    hidden_size = model.vector_size
    print("num of securities : %s, num of dimension : %s" % (len(brand_list), hidden_size))

    doc_arr = np.zeros((len(brand_list), hidden_size))
    for index, name in enumerate(brand_list):
        doc_arr[index, :] = model.docvecs[name]
    return brand_list, doc_arr


def tf_idf(documents, selected_words, brand_list, max_feature = 5000):
    total_freq = Counter()
    corpus = []
    for brand in brand_list:
        review, freq = brand_freq(documents, selected_words, brand)
        total_freq += freq
        doc = ' '.join(review)
        corpus.append(doc)

    total_freq = OrderedDict(sorted(total_freq.items(), key=lambda t: -t[1]))
    vectorizer = TfidfVectorizer(max_features=max_feature)
    tfidf_arr = vectorizer.fit_transform(corpus).toarray()
    col_name = vectorizer.get_feature_names()
    df_tfidf = DataFrame(columns=[col_name])

    for i in range(len(brand_list)):
        df_tfidf.loc[len(df_tfidf)] = tfidf_arr[i]

    df_tfidf.set_index([brand_list], inplace=True)  # 브랜드 이름을 index로
    return df_tfidf




def softmax(x):
    e = np.exp(x - np.max(x,axis=0,keepdims=True))
    x = e / np.sum(e,axis=0,keepdims=True)
    return x

def scoring(model, brand_list, selected, topn=2000):
    # 각 브랜드 이름을 for 돌려서 해당하는 vector 정보를 X 라고 하자
    embedding_size = model.vector_size
    brands_size = len(brand_list)
    X = np.zeros((brands_size, embedding_size))
    for i, brand in enumerate(brand_list):
        X[i] = model.docvecs[brand].flatten()

    # 상위 동사, 부사, 형용사 목록(stopword 포함)
    text = [word for sent in selected for word in sent]
    corpus = nltk.Text(text)
    freq = nltk.FreqDist(corpus)
    top_freq_words = freq.most_common(topn)

    B = X
    m = len(B)
    df_score = DataFrame(columns=[['word'] + brand_list])
    for i, top in enumerate(top_freq_words):
        w = model[top[0]]  # 단어 벡터
        x = np.dot(B, w).reshape((m, 1))
        p = softmax(x)
        # print p.T[0]
        row = [top[0]] + p.T[0].tolist()
        df_score.loc[i] = row
    return df_score

def extract_words_by_score(df_score, brand, documents, selected, min_count = 100):
    keywords = df_score[['word',brand]].sort_values(brand, ascending=False)
    keywords.reset_index(inplace=True, drop=True)
    review, freq = brand_freq(documents, selected, brand)
    df = DataFrame(columns=[["단어","확률유사도","빈도"]])
    for index, row in keywords.iterrows():
        if freq[row['word']] >= min_count:
            df.loc[len(df)] = [row['word'], row[brand], freq[row['word']]]
    return df
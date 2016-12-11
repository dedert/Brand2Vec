from gensim.models import Phrases
from gensim.models.doc2vec import TaggedDocument
import nltk

def prepare(data, col_name ='preprocessed', max_sentence = 50):
    """
    전처리 완료된 문서에서(after raw_preprocess)
    기본적인 자료구조 생성
    :param data: 특정 브랜드(리뷰)의 dataframe, 긍정(4점 이상), 부정(2점 이하)
    :param col_name: 전처리 된 데이터가 들어있는 column name
    :param max_sentence: 한 리뷰당 maximum number of sentence
    :return sentence_list: sentence를 담은 list
    :return sentence_senti_label: 각 sentence 단위의 긍정, 부정 label
    :return numSentence:
    """
    sentence_list = [] #문장단위 list
    review_label = []  # 각 리뷰의 긍, 부정 index
    numSentence = {} #key : doc_index, value : num of sentences in doc_index
    index = 0
    for doc_index, row in data.iterrows():
        if row['overall'] >= 4:
            review_label.append(0)
            if row['sent_length'] >= max_sentence:
                sentence_list.extend(row['preprocessed'][:max_sentence])
                numSentence[doc_index] = max_sentence
                index += len(row[col_name])
            else:
                sentence_list.extend(row['preprocessed'])
                numSentence[doc_index] = len(row[col_name])
                index += len(row[col_name])

        elif row['overall'] <= 2:
            review_label.append(1)
            if row['sent_length'] >= max_sentence:
                sentence_list.extend(row['preprocessed'][:max_sentence])
                numSentence[doc_index] = max_sentence
                index += len(row[col_name])
            else:
                sentence_list.extend(row['preprocessed'])
                numSentence[doc_index] = len(row[col_name])
                index += len(row[col_name])

    return sentence_list, review_label, numSentence

def bigram_and_sentence(sentence_list, review_label, numSentence, max_vocab = 5000, threshold = 10, min_count=5):
    """
    sentence 만 들어있는 list(flatten)를 다시 문서, 문장모양의 list로 변환
    :param sentence_list: 문장 list
    :param numSentence: 각 문서의 문장 길이
    :param threshold: bigram의 threshold
    :return:
    """

    bigram = Phrases(sentences=sentence_list, threshold=threshold, min_count=min_count)
    numDocs = len(numSentence.keys())
    total_token = []
    count = 0
    for i in range(numDocs):
        for num_s in range(numSentence[i]):
            total_token.append(bigram[sentence_list[count]])
            count += 1

    corpus = [word for sentence in total_token for word in sentence]
    text = nltk.Text(corpus)
    freq = nltk.FreqDist(text)
    keywords = [tup[0] for tup in freq.most_common(max_vocab)]

    documents = []
    sentence_list_again = []
    documents_label = []
    count = 0

    for i in range(numDocs):
        doc_list = []
        documents_label.append(review_label[i])
        for num_s in range(numSentence[i]):
            bi = bigram[sentence_list[count]]
            count += 1
            bi = [word for word in bi if word in keywords]  # keywords에 속한 단어만 추출
            if len(bi) >= 1: #문장 길이가 1 이상인 것만 추출
                doc_list.append(bi)
                if review_label[i] == 0:
                    tags = "positive"
                else:
                    tags = "negative"
                document = TaggedDocument(words = bi, tags=[tags])
                documents.append(document)
        sentence_list_again.append(doc_list)
    return documents, sentence_list_again, bigram, documents_label



from preprocess.util import *
import time

class Preprocess:
    def __init__(self, data_path, save_path, category_list):
        self.data_path = data_path
        self.save_path = save_path
        self.raw_data = data_path + "reviews_Electronics.json.gz"
        self.meta_data = data_path + "meta_Electronics.json.gz"
        self.category_list = category_list
        # data_path = "/media/hs-ubuntu/data/dataset/Amazon/" # where data file is located.
        # save_path = "/media/hs-ubuntu/data/dataset/MasterThesis/STMD_data/" # where result file will be saved.
        # category_list = ["Electronics","Beauty","Clothing_Shoes_and_Jewelry"]

    def preprocess(self):
        for category in self.category_list:
            raw_data_path = self.data_path + "reviews_" + category + ".json.gz"
            meta_data_path = self.data_path + "meta_" + category + ".json.gz"

            # print("Start loading %s raw data" % category)
            # start = time.time()
            # raw_data = load_data(raw_data_path, year=2013)
            # end = time.time()
            # print("Completed loading %s raw data, time : %.2f" % (category, end - start))
            #
            #
            # print("Start loading %s meta data", category)
            # start = time.time()
            # meta_data = load_meta(meta_data_path)
            # end = time.time()
            # print("Completed loading %s meta data, time : %.2f" % (category, end - start))
            #
            #
            # print("Start join raw and meta of %s" % category)
            # start = time.time()
            # join_data = join_meta_data(self.raw_data, self.meta_data)
            # end = time.time()
            # print("Completed join raw and meta data of %s, time : %.2f" % (category, end - start))

            # NaN 값을 제거하고, 중간에 값을 저장하기 위해 일단 저장 후 다시 불러옴
            # join_data.to_csv(self.save_path + "join_" + category + ".csv", index=False)
            join_data = pd.read_csv(self.save_path + "join_" + category + ".csv")

            print("Start extract sentences of %s" % category)
            start = time.time()
            join_data = extract_sentence(join_data)
            end = time.time()
            print("Completed extract sentences of %s, time : %.2f" % (category, end - start))


            print("Start extract samples from data")
            start = time.time()
            top_brands_df, top_brands_list = top_brands(join_data)
            data_final = sample_data(top_brands_df, top_brands_list)
            end = time.time()
            print("Completed extract samples of %s, time : %.2f" % (category, end - start))

            print("check shape ----------")
            print("%s shape after sampling : %s, %s"  % (category, data_final.shape[0], data_final.shape[1]))

            # # 중간 저장
            # print("Save before pos tagging start")
            # data_final.to_csv(self.save_path + "raw_" + category + ".csv", index=False)
            # print("Save before pos tagging completed")


            # 형태소 분석
            print("Start pos tag of sentences in %s" % category)
            start = time.time()
            data_final['reviewSentence_tagged'] = data_final.reviewSentence.apply(sentence_postag)
            data_final.to_csv(self.save_path + "pos_tagged_" + category + ".csv", index=False)
            end = time.time()
            print("Completed pos-tagging and save of %s, time : %.2f" % (category, end - start))

            # 형태소 분석한거에다가 추가 전처리
            print("Start preprocessing in %s" % category)
            start = time.time()
            data_final['preprocessed'] = data_final.reviewSentence_tagged.apply(preprocess)
            data_final.to_csv(self.save_path + "preprocess_complete_" + category + ".csv", index=False)
            end = time.time()
            print("Completed preprocess and save of %s, time : %.2f" % (category, end - start))


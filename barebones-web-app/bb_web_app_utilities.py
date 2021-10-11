import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.preprocessing as pp
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 200

RND_SEED = 45822
random.seed(RND_SEED)
np.random.seed(RND_SEED)


def get_demo_data():
    data_df = \
        pd.read_csv("../baseline-classifier/data/consolidated_disaster_tweet_data.tsv", sep="\t")

    return data_df


# def convert_demo_data_into_list(consolidated_disaster_tweet_data_df, limit=50):
#     consolidated_disaster_tweet_data_df["assigned_label"] = "-"
#     consolidated_disaster_tweet_data_df["tweet_id"] = consolidated_disaster_tweet_data_df["tweet_id"].values.astype(str)
#     all_texts = consolidated_disaster_tweet_data_df[["tweet_id", "tweet_text", "assigned_label"]].values.tolist()
#
#     max_length = len(all_texts)
#     if limit < max_length:
#         all_texts_adj = random.sample(all_texts, limit)
#     else:
#         all_texts_adj = all_texts
#
#     return all_texts_adj


def convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df, limit=50):
    consolidated_disaster_tweet_data_df["assigned_label"] = "-"
    consolidated_disaster_tweet_data_df["tweet_id"] = consolidated_disaster_tweet_data_df["tweet_id"].values.astype(str)
    all_texts = consolidated_disaster_tweet_data_df[["tweet_id", "tweet_text", "assigned_label"]].values.tolist()

    max_length = len(all_texts)
    if limit < max_length:
        all_texts_adj = random.sample(all_texts, limit)
    else:
        all_texts_adj = all_texts

    all_texts_json = [{"id": text[0], "text": text[1], "label": text[2]} for text in all_texts_adj]
    return all_texts_json


def update_all_texts(all_texts, text_id, label):
    all_texts_df = pd.DataFrame(all_texts, columns=["tweet_id", "tweet_text", "assigned_label"])
    all_texts_df.loc[all_texts_df["tweet_id"] == str(text_id), "assigned_label"] = label

    all_texts_updated = all_texts_df.values

    return all_texts_updated


def filter_all_texts(all_text, filter_list):
    # filter_list_dict = dict(filter_list)
    # print("all_text :", all_text)
    # print("filter_list :", filter_list)

    filtered_all_text = []

    # Slow - 10,000 records - duration 0:00:31.719903
    # for filter_id in filter_list:
    #     for text in all_text:
    #         if text["id"] == filter_id:
    #             filtered_all_text.append(text)

    # Faster - 10,000 records - duration 0:00:07.619622
    # [filtered_all_text.append(text) for text in all_text if text["id"] in filter_list]

    # Fastest - 10,000 records - duration 0:00:00.102955
    all_text_df = pd.DataFrame(all_text)
    filtered_all_text_df = all_text_df[all_text_df["id"].isin(filter_list)]
    filtered_all_text = filtered_all_text_df.to_dict("records")

    return filtered_all_text


def update_texts_list(texts_list, sub_list_limit, old_obj_lst=[], new_obj_lst=[], texts_list_list=[]):
    print("len(texts_list) :", texts_list)
    updated_texts_list = texts_list # .copy()

    if len(old_obj_lst) > 0 or len(new_obj_lst) > 0:
        if len(old_obj_lst) > 0:
            for old_obj in old_obj_lst:
                print(f"Trying to remove obj : {old_obj}")
                updated_texts_list.remove(old_obj)

        if len(new_obj_lst) > 0:
            for new_obj in new_obj_lst:
                updated_texts_list.append(new_obj)

    texts_list_list.clear()
    updated_texts_list_list = \
        [updated_texts_list[i:i + sub_list_limit] for i in range(0, len(updated_texts_list), sub_list_limit)]
    texts_list_list.extend(updated_texts_list_list)
    print("len(texts_list_list) :", len(texts_list_list))
    return updated_texts_list, updated_texts_list_list


def cosine_similarities(mat):
    # https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
    #
    col_normed_mat = pp.normalize(mat.tocsc(), axis=1)
    return col_normed_mat * col_normed_mat.T


def get_all_similarities(sparse_vectorized_corpus, corpus_text_ids):
    # Slow - vectorized_corpus.shape : (76484, 10) - Unable to allocate 43.6 GiB for an array with shape (76484, 76484) and data type float64
    # similarities = cosine_similarity(sparse_vectorized_corpus)
    # similarities_df = pd.DataFrame(similarities, columns=corpus_text_ids)

    # Faster - vectorized_corpus.shape : (76484, 10) - duration 0:01:43.129781
    # similarities = cosine_similarity(sparse_vectorized_corpus, dense_output=False)
    # similarities_df = pd.DataFrame.sparse.from_spmatrix(similarities, columns=corpus_text_ids)

    # Faster - vectorized_corpus.shape : (76484, 10) - duration 0:02:03.657139
    # similarities = np.dot(sparse_vectorized_corpus, sparse_vectorized_corpus.T)
    # similarities_df = pd.DataFrame.sparse.from_spmatrix(similarities, columns=corpus_text_ids)

    # Fastest - vectorized_corpus.shape : (76484, 10) - duration 0:01:59.331099
    similarities = cosine_similarities(sparse_vectorized_corpus)
    similarities_df = pd.DataFrame.sparse.from_spmatrix(similarities, columns=corpus_text_ids)

    # print("similarities :")
    # print(similarities)

    similarities_df["id"] = corpus_text_ids
    similarities_df = similarities_df.set_index(["id"])
    return similarities_df


def get_all_similarities_one_at_a_time(sparse_vectorized_corpus, corpus_text_ids, text_id, keep_original=False):
    text_id_index = corpus_text_ids.index(text_id)
    # Fastest - vectorized_corpus.shape : (76484, 10) - duration 0:01:59.331099
    single_vectorized_record = sparse_vectorized_corpus[text_id_index, :]
    similarities = np.dot(single_vectorized_record, sparse_vectorized_corpus.T).toarray().ravel()
    similarities_series = pd.Series(similarities, index=corpus_text_ids)
    corpus_text_ids_adj = corpus_text_ids.copy()

    if not keep_original:
        corpus_text_ids_adj.remove(text_id)

    similarities_series = similarities_series.filter(corpus_text_ids_adj)
    similarities_series.index.name = "id"
    similarities_series = similarities_series.sort_values(ascending=False)
    return similarities_series


def __get_all_similarities_one_at_a_time(sparse_vectorized_corpus, corpus_text_ids, text_id, keep_original=False):
    text_id_index = corpus_text_ids.index(text_id)
    # Fastest - vectorized_corpus.shape : (76484, 10) - duration 0:01:59.331099
    single_vectorized_record = sparse_vectorized_corpus[text_id_index, :]
    similarities = np.dot(single_vectorized_record, sparse_vectorized_corpus.T).toarray().ravel()
    similarities_series = pd.Series(similarities, index=corpus_text_ids)
    corpus_text_ids_adj = corpus_text_ids.copy()

    if not keep_original:
        corpus_text_ids_adj.remove(text_id)

    similarities_series = similarities_series.filter(corpus_text_ids_adj)
    similarities_series.index.name = "id"
    similarities_series = similarities_series.sort_values(ascending=False)
    return similarities_series


def get_similarities_single_record(similarities_df, corpus_text_id):
    # keep_indices = [x for x in similarities_df.index.values if x not in [corpus_text_id]]
    # similarities = similarities_df.filter(keep_indices, axis=0)[corpus_text_id].sort_values(ascending=False)

    keep_indices = [x for x in similarities_df.index.values if x not in [corpus_text_id]]
    similarities = similarities_df[corpus_text_id]
    similarities = similarities.filter(keep_indices)
    similarities = similarities.sort_values(ascending=False)


    return similarities


def get_top_similar_texts(all_texts_json, similarities_series, top=5, similar_texts=[]):
    filter_list = similarities_series.head(top).index.values
    top_texts = filter_all_texts(all_texts_json, filter_list)
    similar_texts.clear()
    similar_texts.extend(top_texts)
    return top_texts


if __name__ == "__main__":
    start_time = datetime.now()
    print(">> Start time :", start_time.strftime("%m/%d/%Y %H:%M:%S"), "*"*100)

    consolidated_disaster_tweet_data_df = get_demo_data()
    print("consolidated_disaster_tweet_data_df :")
    print(consolidated_disaster_tweet_data_df.head())
    #
    # all_texts_adj = convert_demo_data_into_list(consolidated_disaster_tweet_data_df, limit=50)
    # print("all_texts_adj :")
    # print(all_texts_adj)
    # print()

    all_texts_json = convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df, limit=100000)
    # print("all_texts_json :")
    # print(all_texts_json)
    # print()

    # print("type(all_texts_adj) :")
    # print(type(all_texts_adj))
    # print()
    # text_id = '798262465234542592'
    # all_texts_updated = update_all_texts(all_texts=all_texts_adj, text_id=text_id, label="Earthquake")
    # print("all_texts_updated :")
    # print(all_texts_updated)
    adj_consolidated_disaster_tweet_data_df = consolidated_disaster_tweet_data_df#.head(50)
    # print("adj_consolidated_disaster_tweet_data_df.head(5)['tweet_text'] :")
    # print(adj_consolidated_disaster_tweet_data_df.head(5)["tweet_text"])

    corpus_text_id = "798262465234542592"

    # print("798262465234542592 :")
    # print(adj_consolidated_disaster_tweet_data_df[adj_consolidated_disaster_tweet_data_df["tweet_id"] == corpus_text_id]["tweet_text"])
    # print()
    # print("721752219201232897 :")
    # print(adj_consolidated_disaster_tweet_data_df[adj_consolidated_disaster_tweet_data_df["tweet_id"] == "798014650231046144"]["tweet_text"])
    # print()
    # print("800600023310286848 :")
    # print(adj_consolidated_disaster_tweet_data_df[adj_consolidated_disaster_tweet_data_df["tweet_id"] == "797905553376899072"]["tweet_text"])
    # print()
    # print("1176515611297533952 :")
    # print(adj_consolidated_disaster_tweet_data_df[adj_consolidated_disaster_tweet_data_df["tweet_id"] == "798277386928328704"]["tweet_text"])
    # print()
    # print("798100160148545536 :")
    # print(adj_consolidated_disaster_tweet_data_df[adj_consolidated_disaster_tweet_data_df["tweet_id"] == "722068424692789248"]["tweet_text"])
    # print()
    # print("910579763537788928 :")
    # print(adj_consolidated_disaster_tweet_data_df[adj_consolidated_disaster_tweet_data_df["tweet_id"] == "798027556041560064"]["tweet_text"])
    # print()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=100)
    vectorized_corpus = \
        vectorizer.fit_transform(adj_consolidated_disaster_tweet_data_df["tweet_text"])

    print("vectorized_corpus.shape :", vectorized_corpus.shape)
    print("type(vectorized_corpus) :", type(vectorized_corpus))
    vectorized_corpus_df = pd.DataFrame.sparse.from_spmatrix(vectorized_corpus, columns=vectorizer.get_feature_names())
    print("vectorized_corpus_df.shape :", vectorized_corpus_df.shape)
    # print("vectorized_corpus_df.head ")
    # print(vectorized_corpus_df.head)

    temp_start_time = datetime.now()
    corpus_text_ids = [str(x) for x in adj_consolidated_disaster_tweet_data_df["tweet_id"].values]
    print(f">> 'corpus_text_ids' calculated @{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} "
          f"duration {datetime.now()-temp_start_time}")
    # print("corpus_text_ids :", corpus_text_ids)

    # ********************************************************************************************************
    temp_start_time = datetime.now()
    similarities_alt = get_all_similarities_one_at_a_time(sparse_vectorized_corpus=vectorized_corpus,
                                                          corpus_text_ids=corpus_text_ids,
                                                          text_id=corpus_text_id)
    print(f">> 'similarities_alt' calculated @{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} "
          f"duration {datetime.now()-temp_start_time}")
    print("type(similarities_alt) :", type(similarities_alt))

    print("similarities_alt :")
    print(similarities_alt.head(5))

    top_5_texts = get_top_similar_texts(all_texts_json=all_texts_json, similarities_series=similarities_alt, top=5)
    print("top_5_texts :")
    print(top_5_texts)
    # ********************************************************************************************************

    # ********************************************************************************************************
    # temp_start_time = datetime.now()
    # similarities_df = get_all_similarities(sparse_vectorized_corpus=vectorized_corpus,
    #                                        corpus_text_ids=corpus_text_ids)
    # print(f">> 'similarities_df' calculated @{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} "
    #       f"duration {datetime.now()-temp_start_time}")
    # print("type(similarities_df) :", type(similarities_df))
    #
    # # print("similarities_df :")
    # # print(similarities_df)
    #
    # temp_start_time = datetime.now()
    # similarities = get_similarities_single_record(similarities_df=similarities_df, corpus_text_id=corpus_text_id)
    # print(f">> 'similarities' calculated @{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} "
    #       f"duration {datetime.now()-temp_start_time}")
    # print("type(similarities) :", type(similarities))
    #
    # print("similarities :")
    # print(similarities.head(5))
    #
    # filter_list = list(similarities.index)
    # # print("filter_list :", filter_list)
    #
    # temp_start_time = datetime.now()
    # filtered_all_text = filter_all_texts(all_text=all_texts_json, filter_list=filter_list)
    # print(f">> 'filtered_all_text' calculated @{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} "
    #       f"duration {datetime.now()-temp_start_time}")
    # # print("filtered_all_text :")
    # # print(filtered_all_text)
    # ********************************************************************************************************

    # ********************************************************************************************************
    temp_start_time = datetime.now()
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    y_classes = ["Earthquake", "Fire", "Flood", "Hurricane"]

    print("all_texts_json :")
    print(all_texts_json[:20])
    # X_train =
    # y_train =
    # y_pred =
    # clf.fit(X_train, y_train)
    #
    labels_dict = {
        "798262465234542592": "Earthquake",
        "771464543796985856": "Earthquake",
        "": "",
    }
    #
    # print(f">> 'similarities_alt' calculated @{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} "
    #       f"duration {datetime.now()-temp_start_time}")
    # ********************************************************************************************************
    end_time = datetime.now()
    duration = end_time - start_time

    print(">> End time :", end_time.strftime("%m/%d/%Y @ %H:%M:%S"), "*"*100)
    print(">> Duration :", duration, "*"*100)




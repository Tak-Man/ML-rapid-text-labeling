from flask import Flask, render_template, request, jsonify, flash
import bb_web_app_utilities as utils
from sklearn.feature_extraction.text import TfidfVectorizer
import copy
# from flask.ext.session import Session


app = Flask(__name__)
app.secret_key = "super secret key"
app.config.from_object(__name__)
# Session(app)

TEXTS_LIMIT = 100000
TABLE_LIMIT = 50
MAX_FEATURES = 100

consolidated_disaster_tweet_data_df = utils.get_demo_data()
# ALL_TEXTS = utils.convert_demo_data_into_list(consolidated_disaster_tweet_data_df, limit=TEXTS_LIMIT)
TEXTS_LIST = utils.convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df, limit=TEXTS_LIMIT)
TEXTS_LIST_LIST = [TEXTS_LIST[i:i + TABLE_LIMIT] for i in range(0, len(TEXTS_LIST), TABLE_LIMIT)]
TOTAL_PAGES = len(TEXTS_LIST_LIST)
print("TOTAL_PAGES :", TOTAL_PAGES)
TEXTS_GROUP_1 = []
TEXTS_GROUP_2 = []

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=MAX_FEATURES)
VECTORIZED_CORPUS = \
    vectorizer.fit_transform(consolidated_disaster_tweet_data_df["tweet_text"])

CORPUS_TEXT_IDS = [str(x) for x in consolidated_disaster_tweet_data_df["tweet_id"].values]
# print("type(vectorized_corpus) :", type(VECTORIZED_CORPUS))
# print("vectorized_corpus.shape :", VECTORIZED_CORPUS.shape)


@app.route("/")
def index():
    page_number = 0
    return render_template("text_labeling_1.html",
                           selected_text_id="None",
                           selected_text="Select a text below to begin labeling.",
                           info_message="",
                           page_number=0,
                           total_pages=TOTAL_PAGES,
                           texts_list=TEXTS_LIST_LIST[page_number],
                           texts_group_1=[],
                           texts_group_2=[])


@app.route("/text_labeling_1.html", methods=["GET", "POST"])
def text_labeling():

    selected_text_id = request.args.get("selected_text_id", None)
    selected_text = request.args.get("selected_text", None)
    page_number = int(request.args.get("page_number", None))
    # print("selected_text_id :", selected_text_id)
    # print("selected_text :", selected_text)
    # print("type(page_number) :", type(page_number))

    similarities_series = utils.get_all_similarities_one_at_a_time(sparse_vectorized_corpus=VECTORIZED_CORPUS,
                                                                   corpus_text_ids=CORPUS_TEXT_IDS,
                                                                   text_id=selected_text_id,
                                                                   keep_original=False)

    utils.get_top_similar_texts(all_texts_json=TEXTS_LIST,
                                 similarities_series=similarities_series, top=5, similar_texts=TEXTS_GROUP_1)
    print("len(TEXTS_GROUP_1) :", len(TEXTS_GROUP_1))

    return render_template("text_labeling_1.html",
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           info_message="Text selected",
                           page_number=page_number,
                           total_pages=TOTAL_PAGES,
                           texts_list=TEXTS_LIST_LIST[page_number],
                           texts_group_1=TEXTS_GROUP_1,
                           texts_group_2=TEXTS_GROUP_2)


@app.route("/go_to_page", methods=["GET", "POST"])
def go_to_page():
    page_number = int(request.args.get("page_number", None))
    print("page_number :", page_number)
    return render_template("text_labeling_1.html",
                           selected_text_id="None",
                           selected_text="Select a text below to begin labeling.",
                           info_message="",
                           page_number=page_number,
                           total_pages=TOTAL_PAGES,
                           texts_list=TEXTS_LIST_LIST[page_number],
                           texts_group_1=[],
                           texts_group_2=[])


@app.route("/single_text", methods=["GET", "POST"])
def single_text():
    if request.method == "GET":
        if len(TEXTS_LIST) > 0:
            # return jsonify(TEXTS_LIST)
            id = request.form["selected_text_id"]
            text = request.form["selected_text"]
            return render_template("text_labeling_1.html",
                                   selected_text_id=id,
                                   selected_text=text,
                                   info_message="GET",
                                   texts_list=jsonify(TEXTS_LIST),
                                   texts_group_1=TEXTS_GROUP_1,
                                   texts_group_2=TEXTS_GROUP_2)
        else:
            "Nothing Found", 404

    if request.method == "POST":
        print(f"selected_label_single : '{request.form.get('selected_label_single')}'")
        new_id = request.form["selected_text_id"]
        new_text = request.form["selected_text"]
        page_number = int(request.form["page_number"])
        new_label = request.form["selected_label_single"]
        info_message = ""

        if new_label == "" or new_id == "None":
            if new_label == "":
                flash("Select a 'Label'.", "error")
                info_message += f"Select a 'Label'."

            if new_id == "None":
                info_message += "\n" + f"Select a 'Text ID'."

            return render_template("text_labeling_1.html",
                                   selected_text_id=new_id,
                                   selected_text=new_text,
                                   info_message=info_message,
                                   page_number=page_number,
                                   total_pages=TOTAL_PAGES,
                                   texts_list=TEXTS_LIST_LIST[page_number],
                                   texts_group_1=TEXTS_GROUP_1,
                                   texts_group_2=TEXTS_GROUP_2)
        else:


            old_obj = {"id": new_id, "text": new_text, "label": "-"}
            new_obj = {"id": new_id, "text": new_text, "label": new_label}

            utils.update_texts_list(texts_list=TEXTS_LIST,
                                    sub_list_limit=TABLE_LIMIT,
                                    old_obj_lst=[old_obj],
                                    new_obj_lst=[new_obj],
                                    texts_list_list=TEXTS_LIST_LIST)

            return render_template("text_labeling_1.html",
                                   selected_text_id=new_id,
                                   selected_text=new_text,
                                   info_message="Label assigned",
                                   page_number=page_number,
                                   total_pages=TOTAL_PAGES,
                                   texts_list=TEXTS_LIST_LIST[page_number],
                                   texts_group_1=TEXTS_GROUP_1,
                                   texts_group_2=TEXTS_GROUP_2)


@app.route("/grouped_texts", methods=["GET", "POST"])
def grouped_texts():
    print("Trying to assign label to group...")
    print("request.method :", request.method)
    # if request.method == "GET":
    #     if len(TEXTS_LIST) > 0:
    #         # return jsonify(TEXTS_LIST)
    #         id = request.form["selected_text_id"]
    #         text = request.form["selected_text"]
    #
    #
    #         return render_template("text_labeling_1.html",
    #                                selected_text_id=id,
    #                                selected_text=text,
    #                                info_message="GET",
    #                                texts_list=jsonify(TEXTS_LIST),
    #                                texts_group_1=[],
    #                                texts_group_2=[])
    #     else:
    #         "Nothing Found", 404

    if request.method == "POST":
        # texts_group_1 = request.form["texts_group_1"]
        print("TEXTS_GROUP_1 :")
        print(TEXTS_GROUP_1)
        print()

        page_number = int(request.form["page_number"])
        print("page_number :", page_number)
        print()

        new_id = request.form["selected_text_id"]
        print("new_id :", new_id)
        print()

        new_text = request.form["selected_text"]
        print("new_text :", new_text)
        print()

        # new_label = request.form["assigned_label_group"]
        # print("new_label :", new_label)
        # print()

        new_label = request.form["selected_label_group1"]
        print("new_label :", new_label)
        print()

        info_message = ""
        if new_label == "" or len(TEXTS_GROUP_1) == 0:
            if new_label == "":
                info_message += f"Select a 'Label'."

            if len(TEXTS_GROUP_1) == 0:
                info_message += "\n" + f"Select a 'Text ID'."

            return render_template("text_labeling_1.html",
                                   selected_text_id=new_id,
                                   selected_text=new_text,
                                   info_message=info_message,
                                   page_number=page_number,
                                   total_pages=TOTAL_PAGES,
                                   texts_list=TEXTS_LIST_LIST[page_number],
                                   texts_group_1=TEXTS_GROUP_1,
                                   texts_group_2=[])

        else:
            texts_group_1_updated = copy.deepcopy(TEXTS_GROUP_1)
            for obj in texts_group_1_updated:
                obj["label"] = new_label

            print("texts_group_1_updated :")
            print(texts_group_1_updated)

            utils.update_texts_list(texts_list=TEXTS_LIST,
                                    sub_list_limit=TABLE_LIMIT,
                                    old_obj_lst=TEXTS_GROUP_1,
                                    new_obj_lst=texts_group_1_updated,
                                    texts_list_list=TEXTS_LIST_LIST)
            print("Completed 'update_texts_list'")
            print("Trying to render template for groups assignment...")
            return render_template("text_labeling_1.html",
                                   selected_text_id=new_id,
                                   selected_text=new_text,
                                   info_message="Labels assigned to group",
                                   page_number=page_number,
                                   total_pages=TOTAL_PAGES,
                                   texts_list=TEXTS_LIST_LIST[page_number],
                                   texts_group_1=texts_group_1_updated,
                                   texts_group_2=[])


@app.route("/text/<id>", methods=["GET", "PUT", "DELETE"])
def single_record(id):
    if request.method == "GET":
        for text in TEXTS_LIST:
            if text in TEXTS_LIST:
                return jsonify(text)
            pass

    if request.method == "PUT":
        for text in TEXTS_LIST:
            if text["id"] == id:
                text["text"] = request.fomr["selected_text"]
                text["label"] = request.form["assigned_label"]
                updated_text = {"id": id, "text": text["text"], "label": text["label"]}
                return jsonify(updated_text)

    if request.method == "DELETE":
        for index, book in enumerate(TEXTS_LIST):
            if text["id"] == id:
                TEXTS_LIST.pop(index)
                return jsonify(TEXTS_LIST)


if __name__ == "__main__":
    app.run()


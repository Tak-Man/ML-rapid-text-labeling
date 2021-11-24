from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
from sklearn.metrics import accuracy_score
import os
import glob
from time import sleep
import datetime
import pickle
from zipfile import ZipFile
import sys

sys.path.insert(1, '../baseline-classifier/utilities')
import dt_utilities as utils

def get_radio_buttons(driver):
    radio_buttons = []
    radio_buttons.append(driver.find_element_by_id('category1'))
    radio_buttons.append(driver.find_element_by_id('category2'))
    radio_buttons.append(driver.find_element_by_id('category3'))
    radio_buttons.append(driver.find_element_by_id('category4'))

    return radio_buttons


def select_label_one_text(driver, radio_button_id, wait_time=0.75):
    # we select the correct radio button
    radio_buttons = get_radio_buttons(driver)
    sleep(wait_time)
    radio_buttons[radio_button_id].click()
    # label one example
    button_label_single = driver.find_element_by_id('labelButtonSingle')
    button_label_single.click()
    sleep(wait_time)


def select_label_multi_text(driver, xpath, op_base_xpath, radio_button_id, wait_time=0.75, max_options=1, label_type="SimilarTexts",
                            min_recommender_labels=1000, click_needed=True, true_labeling_cutoff=0,
                            true_labeling_cutoff_end=0):
    total_unlabeled, total_labeled = get_total_unlabeled(driver, get_labeled=True)
    if click_needed:
        # select a text from the list of all texts
        driver.find_element_by_xpath(xpath).click()
        sleep(wait_time)
    # we select the correct radio button
    radio_buttons = get_radio_buttons(driver)
    sleep(wait_time)
    # print(radio_button_id)
    radio_buttons[radio_button_id].click()
    #
    op_xpath = op_base_xpath + str(max_options) + ']'
    # print(op_xpath)
    driver.find_element_by_xpath(op_xpath).click()
    # label multi example
    if (total_labeled < true_labeling_cutoff) | (total_unlabeled < true_labeling_cutoff_end):
        button_label_ten = driver.find_element_by_id('labelButtonSingle')
    elif (label_type == "SimilarTexts") | (total_labeled < min_recommender_labels) | (
            total_unlabeled < min_recommender_labels):
        driver.find_element_by_id("buttonSimilarTexts1Buttons").click()
        button_label_ten = driver.find_element_by_id('group1Button')
    elif label_type == "RecommendedTexts":
        driver.find_element_by_id("buttonSimilarTexts2Buttons").click()
        button_label_ten = driver.find_element_by_id('group2Button')
    sleep(wait_time)
    button_label_ten.click()

def click_difficult_texts(driver, wait_time=0.75):
    sleep(wait_time)
    button_difficult_texts = driver.find_element_by_id('generateDifficultTextsButton')
    button_difficult_texts.click()

def scroll_label_ten(driver, radio_button_id, scroll_wait_seconds=1.75):
    # we scroll down the results list
    for scr in range(2, 10, 2):
        scr_xpath = '//*[@id="group1Table"]/tbody/tr[' + str(scr) + ']/td[1]/a'
        print(scr_xpath)
        link_scroll = driver.find_element_by_xpath(scr_xpath)
        driver.execute_script("return arguments[0].scrollIntoView(true);", link_scroll)
        sleep(scroll_wait_seconds)
    radio_buttons = get_radio_buttons()
    radio_buttons[radio_button_id].click()
    sleep(scroll_wait_seconds)

    # we apply a group label after checking all 10 suggested are correct
    button_label_ten = driver.find_element_by_id('group1Button')
    sleep(scroll_wait_seconds)
    button_label_ten.click()


def get_total_unlabeled(driver, get_labeled=False):
    total_unlabeled = driver.find_element_by_xpath('//*[@id="summarySectionTable"]/tbody/tr[3]/td[2]').text.replace(',',
                                                                                                                    '')
    total_unlabeled = int(total_unlabeled)
    if get_labeled:
        total_labeled = driver.find_element_by_xpath('//*[@id="summarySectionTable"]/tbody/tr[4]/td[2]').text.replace(
            ',', '')
        total_labeled = int(total_labeled)
        return total_unlabeled, total_labeled
    else:
        return total_unlabeled

def get_overall_quality_score(driver):
    overall_quality_score = driver.find_element_by_xpath('//*[@id="difficultTextSummaryTable"]/tbody/tr[1]/td[2]').text
    return overall_quality_score

def export_model(driver):
    driver.find_element_by_id("exportRecordsButton").click()
    # Create a ZipFile Object and load sample.zip in it
    for n in range(10000):
        try:
            results_zip = glob.glob(os.path.join("models", "*.zip"))
            mresults = results_zip[0]
            #print(mresults)
            with ZipFile(mresults, 'r') as zipObj:
               # Extract all the contents of zip file in current directory
               zipObj.extractall()
            break
        except:
            sleep(0.01)

def get_accuracy_score(test_df, vectorizer_needs_transform):
    # load the model from disk
    model_filename = os.path.join("output", "trained-classifier.pkl")
    loaded_model = pickle.load(open(model_filename, 'rb'))
    if vectorizer_needs_transform:
        vectorizer_filename = os.path.join("output", "fitted-vectorizer.pkl")
        vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
        X_test = vectorizer.transform(test_df["tweet_text"])
    y_test = test_df["event_type"]
    y_pred = [x.lower() for x in loaded_model.predict(X_test)]
    test_accuracy_score = accuracy_score(y_test, y_pred)

    return test_accuracy_score, vectorizer_needs_transform

def clear_model_output():
    filelist = glob.glob(os.path.join("models", "*.zip"))
    for f in filelist:
        os.remove(f)

def clear_output():
    filelist = glob.glob(os.path.join("output", "*.*"))
    for f in filelist:
        os.remove(f)

def get_tracker_row(driver, test_df, starttime, vectorizer_needs_transform, fully_human_labeled=True):
    try:
        overall_quality_score = get_overall_quality_score(driver)
    except:
        overall_quality_score = 0.
    _, total_labeled = get_total_unlabeled(driver, get_labeled=True)

    test_accuracy_score = 0.

    try:
        clear_model_output()
        clear_output()
        export_model(driver)
        test_accuracy_score, vectorizer_needs_transform = get_accuracy_score(test_df, vectorizer_needs_transform)
    except:
        pass

    currenttime = datetime.datetime.now()
    elapsedtime = currenttime - starttime

    tracker_row = {'labels': total_labeled,
                   'overall_quality_score': overall_quality_score,
                   'accuracy': test_accuracy_score,
                   'elapsed_time': elapsedtime,
                   'fully_human_labeled': fully_human_labeled
                   }

    return tracker_row, vectorizer_needs_transform

def get_label_id(label):
    radio_button_id = "other"
    if label == str.lower("earthquake"):
        radio_button_id = 0
    if label == str.lower("fire"):
        radio_button_id = 1
    if label == str.lower("flood"):
        radio_button_id = 2
    if label == str.lower("hurricane"):
        radio_button_id = 3
    return radio_button_id

def get_true_label(driver, train_df, tweet_id=-1):
    # to mimic full single label at a time human labeling
    tweet_id = int(tweet_id)
    if tweet_id == -1:
        tweet_text = driver.find_element_by_xpath('//*[@id="currentTextBody"]/p').text
        tweet_text = str.lower(tweet_text)
        true_label = train_df.loc[train_df['tweet_text_lower'].str.contains(tweet_text), "event_type"].values[0]
    else:
        true_label = train_df.loc[train_df['tweet_id'] == tweet_id, "event_type"].values[0]

    return true_label

def get_true_label_id(driver, train_df, tweet_id=-1):
    true_label = get_true_label(driver, train_df, tweet_id=tweet_id)
    true_label_id = get_label_id(true_label)
    return true_label_id

def process_true_labeling(driver, train_df, test_df, true_labels, starttime, df_tracker, vectorizer_needs_transform, tweet_id=-1):
    true_label_id = get_true_label_id(driver, train_df, tweet_id=tweet_id)  # to mimic full single label at a time human labeling
    # print(true_label_id)
    select_label_one_text(driver, true_label_id, wait_time=0.)
    true_labels += 1
    label_applied = True
    if label_applied == True:
        click_difficult_texts(driver)
    tracker_row, vectorizer_needs_transform = get_tracker_row(driver, test_df, starttime, vectorizer_needs_transform, fully_human_labeled=True)
    print(tracker_row)
    df_tracker = df_tracker.append(tracker_row, ignore_index=True)

    return true_labels, df_tracker

def get_select_tweet_xpath(rrow_, txts_per_page_):
    if rrow_ <= txts_per_page_:
        select_tweet_xpath = '//*[@id="allTextsTable"]/tbody/tr[' + str(rrow_) + ']/td[1]'
    else:
        print("Label from Difficult Texts")
        select_tweet_xpath = '//*[@id="difficultTextsTable"]/tbody/tr[' + str(rrow_ - txts_per_page_) + ']/td[1]'

    return select_tweet_xpath


def search_exclude_labeling(driver, test_df, starttime, df_test_data, vectorizer_needs_transform, wait_time=0.75):
    df_tracker = pd.DataFrame(columns=['labels', 'overall_quality_score', 'accuracy', 'elapsed_time'])
    print(len(df_test_data))

    for rrow in range(len(df_test_data)):
        row_include = df_test_data.loc[df_test_data.index == rrow, "include"].values[0]
        row_exclude = df_test_data.loc[df_test_data.index == rrow, "exclude"].values[0]
        # print(row_exclude)

        supplied_label = df_test_data.loc[df_test_data.index == rrow, "label"].values[0]
        radio_button_id = get_label_id(supplied_label)
        # print(supplied_label, radio_button_id)

        # print("send include", rrow)
        element_include = driver.find_element_by_id('searchAllTextsInclude')
        element_include.send_keys(row_include)

        # print("send exclude", rrow)
        if type(row_exclude) == str:
            element_exclude = driver.find_element_by_id('searchAllTextsExclude')
            element_exclude.send_keys(row_exclude)

        # print("run query", rrow)
        driver.find_element_by_id('searchAllTextsButton').click()
        sleep(wait_time)

        # print("select label option", radio_button_id)
        radio_buttons = get_radio_buttons(driver)
        radio_buttons[radio_button_id].click()
        sleep(wait_time)

        # print("apply label", rrow)
        driver.find_element_by_id('labelSearchTextsButton').click()
        sleep(wait_time)

        # print("update difficult texts / results", rrow)
        click_difficult_texts(driver)
        sleep(wait_time)

        tracker_row, vectorizer_needs_transform = get_tracker_row(driver, test_df, starttime, vectorizer_needs_transform, fully_human_labeled=False)
        df_tracker = df_tracker.append(tracker_row, ignore_index=True)
        print(tracker_row)

    return df_tracker

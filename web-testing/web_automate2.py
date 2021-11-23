# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 08:58:37 2021

@author: michp-ai
"""

# This script is web automation for the Capstone project on ML rapid text labeling
# Before running this script in a different console start the web server by running main.py for the web app
# This is a simple demo script to illustrate how selenium interacts with the web app

# %%
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
from utilities import get_radio_buttons, select_label_one_text, select_label_multi_text, click_difficult_texts, scroll_label_ten, get_total_unlabeled, get_overall_quality_score
from utilities import export_model, get_accuracy_score, clear_model_output, clear_output, get_tracker_row, get_label_id, get_true_label, get_true_label_id, process_true_labeling, get_select_tweet_xpath

# %%
# set a timer
starttime = datetime.datetime.now()

# %%
# Get the data we'll need for evaluation
consolidated_disaster_tweet_data_df = \
    utils.get_consolidated_disaster_tweet_data(root_directory="../baseline-classifier/data/",
                                               event_type_directory="HumAID_data_event_type",
                                               events_set_directories=["HumAID_data_events_set1_47K",
                                                                       "HumAID_data_events_set2_29K"],
                                               include_meta_data=True)

train_df = consolidated_disaster_tweet_data_df[consolidated_disaster_tweet_data_df["data_type"] == "train"].reset_index(
    drop=True)
train_df['tweet_text_lower'] = train_df['tweet_text'].str.lower()
test_df = consolidated_disaster_tweet_data_df[consolidated_disaster_tweet_data_df["data_type"] == "test"].reset_index(
    drop=True)
vectorizer_needs_transform = True

# %%
download_dir = os.path.join(os.getcwd(), "models")
chrome_options = Options()
chrome_options.add_experimental_option('prefs', {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
}
                                       )

# %%
# PARAMETERS
cwd = os.getcwd()
mpath = os.path.join(cwd, "chromedriver.exe")
wait_time = 0  # .75 #0.75
scroll_wait_seconds = 0  # 1.75 #1.75

# %%
driver = webdriver.Chrome(mpath, options=chrome_options)

# %%
# load the webpage
driver.get("http://127.0.0.1:5000/")
driver.maximize_window()
# sleep(2) #for demo

# %%
# navigate landing page
driver.find_element_by_xpath('//*[@id="bodyLeftTable1"]/tbody/tr[1]/td[1]/a').click()
driver.find_element_by_id('config1').click()
driver.find_element_by_id('loadDataSetButton').click()


# %%
# identify radio buttons


# %%
# Set up accuracy tracker
df_tracker = pd.DataFrame(
    columns=['labels', 'overall_quality_score', 'accuracy', 'elapsed_time', 'fully_human_labeled'])

# PARAMETERS
initial_true_label_pages = 1000
initial_true_label_labels_per_page = 50
true_labeling_cutoff = 4000 # 5000
true_labeling_cutoff_end = 0 # 2000

label_type = "RecommendedTexts"  # list of valid values ["SimilarTexts", "RecommendedTexts"]
min_recommender_labels = 3000 #1000  # 3000
display_options_list = [2, 5, 10, 20, 50, 100]
max_display_options = 0
display_option_upper_limit = 6
txts_per_page = 50  # 50
pages_per_max_display_option = 80 #071  # 1071 #1071

difficult_texts_per_page = 10  # need to add functionality for this per page on auto-labeling (& manual labeling per page)

label_applied = False

# %%
# Label single labels
sectionstarttime = datetime.datetime.now()

true_labels = 0

if true_labeling_cutoff > 0:
    for pg in range(initial_true_label_pages):
        if get_total_unlabeled(driver) == 0:
            break
            # loop through page
        for rrow in range(1, initial_true_label_labels_per_page + difficult_texts_per_page + 1):
            if get_total_unlabeled(driver) == 0:
                break
            select_tweet_xpath = get_select_tweet_xpath(rrow, txts_per_page)
            tweet_selected = driver.find_element_by_xpath(select_tweet_xpath)
            tweet_id = tweet_selected.text
            tweet_selected.click()
            # label based on text contents
            if get_total_unlabeled(driver) == 0:
                break
            #try:
            true_labels, df_tracker = process_true_labeling(driver, train_df, test_df, true_labels, starttime, df_tracker, vectorizer_needs_transform,
                                                                tweet_id=tweet_id)
            #except:
            #print("tweet not found single")
            #    pass

            if true_labels >= true_labeling_cutoff:
                print("reached true labeling cutoff inner")
                break

        if true_labels >= true_labeling_cutoff:
            print("reached true labeling cutoff outer")
            break

        # go to next page
        driver.find_element_by_xpath('//*[@id="allTextTableNextButtons"]/a[6]').click()

        # go to next page
    driver.find_element_by_xpath('//*[@id="allTextTableNextButtons"]/a[6]').click()

sectionendtime = datetime.datetime.now()
elapsedsectiontime = sectionendtime - sectionstarttime
print("Elapsed section time", elapsedsectiontime)

# %%
auto_start_page = int(driver.find_element_by_xpath('//*[@id="allTextTableNextButtons"]/a[3]').text)
print(auto_start_page)

# %%
# batch labeling starts
sectionstarttime = datetime.datetime.now()

for op in range(len(display_options_list)):
    # check how many are unlabelled
    if get_total_unlabeled(driver) == 0:
        break
    if label_type == "SimilarTexts":
        op_base_xpath = '//*[@id="group1_table_limit"]/option['
    elif label_type == "RecommendedTexts":
        op_base_xpath = '//*[@id="group2_table_limit"]/option['

    for pg in range(auto_start_page, pages_per_max_display_option + auto_start_page):
        print("page", pg)
        # check how many are unlabelled
        if get_total_unlabeled(driver) == 0:
            break
            # loop through page
        for rrow in range(1, txts_per_page + difficult_texts_per_page + 1):
            # check how many are unlabelled
            if get_total_unlabeled(driver) == 0:
                break
            select_tweet_xpath = get_select_tweet_xpath(rrow, txts_per_page)
            tweet_selected = driver.find_element_by_xpath(select_tweet_xpath)
            tweet_id = tweet_selected.text
            tweet_selected.click()
            # label based on text contents
            if get_total_unlabeled(driver) == 0:
                break
            try:
                true_label_id = get_true_label_id(tweet_id=tweet_id)
                # print(true_label_id)
                select_label_multi_text(select_tweet_xpath + '/a', true_label_id, wait_time=wait_time,
                                        max_options=op+1,
                                        label_type=label_type, min_recommender_labels=min_recommender_labels,
                                        click_needed=False,
                                        true_labeling_cutoff=true_labeling_cutoff,
                                        true_labeling_cutoff_end=true_labeling_cutoff_end)
                label_applied = True
                if label_applied == True:
                    click_difficult_texts()
                tracker_row, vectorizer_needs_transform = get_tracker_row(vectorizer_needs_transform)
                print(tracker_row)
                df_tracker = df_tracker.append(tracker_row, ignore_index=True)
            except:
                print("tweet not found batch")
                pass

                # go to next page
        driver.find_element_by_xpath('//*[@id="allTextTableNextButtons"]/a[6]').click()

sectionendtime = datetime.datetime.now()
elapsedsectiontime = sectionendtime - sectionstarttime
print("Elapsed section time", elapsedsectiontime)

# %%
df_tracker.to_csv("tracker_output.csv")
print(df_tracker.head(20))
print(df_tracker.tail(20))

# %%
driver.close()

# %%
endtime = datetime.datetime.now()
elapsedtime = endtime - starttime
print("Elapsed time", elapsedtime)

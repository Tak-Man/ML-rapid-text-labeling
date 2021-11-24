# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 08:58:37 2021

@author: michp-ai
"""

# This script is web automation for the Capstone project on ML rapid text labeling
# Before running this script in a different console start the web server by running main.py for the web app
# This is a simple demo script to illustrate how selenium interacts with the web app

#%%
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import re
import string
import os
import glob, os.path
from time import sleep
import datetime
import pickle
from zipfile import ZipFile
import sys
sys.path.insert(1, '../baseline-classifier/utilities')
import dt_utilities as utils
import math
from utilities import search_exclude_labeling, get_radio_buttons, select_label_one_text, select_label_multi_text, click_difficult_texts, scroll_label_ten, get_total_unlabeled, get_overall_quality_score
from utilities import export_model, get_accuracy_score, clear_model_output, clear_output, get_tracker_row, get_label_id, get_true_label, get_true_label_id, process_true_labeling, get_select_tweet_xpath

#%%
#set a timer
starttime = datetime.datetime.now()

#%%
# Get the data we'll need for evaluation
consolidated_disaster_tweet_data_df = \
    utils.get_consolidated_disaster_tweet_data(root_directory="../baseline-classifier/data/",
                                               event_type_directory="HumAID_data_event_type",
                                               events_set_directories=["HumAID_data_events_set1_47K",
                                                                       "HumAID_data_events_set2_29K"],
                                               include_meta_data=True)

train_df = consolidated_disaster_tweet_data_df[consolidated_disaster_tweet_data_df["data_type"]=="train"].reset_index(drop=True)
test_df = consolidated_disaster_tweet_data_df[consolidated_disaster_tweet_data_df["data_type"]=="test"].reset_index(drop=True)
vectorizer_needs_transform = True


#%%
download_dir = os.path.join(os.getcwd(), "models")
chrome_options = Options()
chrome_options.add_experimental_option('prefs',  {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    }
)

#%%
# PARAMETERS
mpath = os.getcwd() + "\chromedriver.exe"
wait_time = 0#.75 #0.75
scroll_wait_seconds = 0#1.75 #1.75


#%%
driver = webdriver.Chrome(mpath, options = chrome_options)

#%%
# load the webpage
driver.get("http://127.0.0.1:5000/")
driver.maximize_window()
#sleep(2) #for demo

#%%
# navigate landing page
driver.find_element_by_xpath('//*[@id="bodyLeftTable1"]/tbody/tr[1]/td[1]/a').click()
driver.find_element_by_id('config1').click()
driver.find_element_by_id('loadDataSetButton').click()

#%%
# identify radio buttons


#%%
# read the contents of the text
sectionstarttime = datetime.datetime.now()
#
label_type = "AllTexts_search_exclude" # list of valid values ["SimilarTexts", "RecommendedTexts"]

#print(len(df_test_data))
df_test_data = pd.read_csv("test_data_search_exclude.csv")
clear_model_output()
clear_output()
df_tracker = search_exclude_labeling(driver, test_df, starttime, df_test_data, vectorizer_needs_transform)

sectionendtime = datetime.datetime.now()
elapsedsectiontime = sectionendtime - sectionstarttime 
print("Elapsed section time", elapsedsectiontime)

#%%

#df_test_data = pd.read_csv("test_data_search_exclude2.csv")
#df_tracker = search_exclude_labeling(driver, df_test_data, vectorizer_needs_transform)


#%%
df_tracker.to_csv("tracker_output.csv")
print(df_tracker.head(20))
print(df_tracker.tail(20))

#%%
#driver.close()

#%%
endtime = datetime.datetime.now()
elapsedtime = endtime - starttime 
print("Elapsed time", elapsedtime)
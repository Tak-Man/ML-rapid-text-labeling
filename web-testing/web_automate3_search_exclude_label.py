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
from selenium.webdriver.chrome.options import Options
import pandas as pd
import glob, os.path
import datetime
import sys
sys.path.insert(1, '../baseline-classifier/utilities')
import dt_utilities as utils
from utilities import search_exclude_labeling
from utilities import label_all, clear_model_output, clear_output

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
sectionstarttime = datetime.datetime.now()
label_type = "AllTexts_search_exclude" # list of valid values ["SimilarTexts", "RecommendedTexts, AllTexts_search_exclude"]

df_test_data = pd.read_csv("test_data_search_exclude_very_short.csv")
clear_model_output()
clear_output()
df_tracker = search_exclude_labeling(driver, test_df, starttime, df_test_data, vectorizer_needs_transform)

# finish by labeling all remaining unlabeled examples
df_tracker = label_all(driver, test_df, starttime, df_tracker, vectorizer_needs_transform)

sectionendtime = datetime.datetime.now()
elapsedsectiontime = sectionendtime - sectionstarttime 
print("Elapsed section time", elapsedsectiontime)

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
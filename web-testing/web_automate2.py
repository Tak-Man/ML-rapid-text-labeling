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
from time import sleep
import datetime
import pickle
from zipfile import ZipFile
import sys
sys.path.insert(1, '../baseline-classifier/utilities')
import dt_utilities as utils


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
train_df['tweet_text_lower'] = train_df['tweet_text'].str.lower()
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
def get_radio_buttons():
    radio_buttons = []
    radio_buttons.append(driver.find_element_by_id('category1'))
    radio_buttons.append(driver.find_element_by_id('category2'))
    radio_buttons.append(driver.find_element_by_id('category3'))
    radio_buttons.append(driver.find_element_by_id('category4'))
    
    return radio_buttons

def select_label_one_text(radio_button_id, wait_time=0.75):
    # we select the correct radio button    
    radio_buttons = get_radio_buttons()
    sleep(wait_time)
    radio_buttons[radio_button_id].click() 
    # label one example
    button_label_single = driver.find_element_by_id('labelButtonSingle')
    button_label_single.click()
    sleep(wait_time)
    
def select_label_multi_text(xpath, radio_button_id, wait_time=0.75, max_options=1, label_type="SimilarTexts",
                            min_recommender_labels=1000):
    total_unlabeled, total_labeled = get_total_unlabeled(get_labeled=True)
    # select a text from the list of all texts
    driver.find_element_by_xpath(xpath).click() 
    sleep(wait_time)
    # we select the correct radio button    
    radio_buttons = get_radio_buttons()
    sleep(wait_time)
    radio_buttons[radio_button_id].click() 
    #
    op_xpath = op_base_xpath + str(max_options) + ']'
    driver.find_element_by_xpath(op_xpath).click()
    # label multi example
    if (label_type=="SimilarTexts") | (total_labeled < min_recommender_labels) | (total_unlabeled < min_recommender_labels):
        driver.find_element_by_id("buttonSimilarTexts1Buttons").click()
        button_label_ten = driver.find_element_by_id('group1Button')
    elif label_type=="RecommendedTexts":
        driver.find_element_by_id("buttonSimilarTexts2Buttons").click()
        button_label_ten = driver.find_element_by_id('group2Button')
    sleep(wait_time)
    button_label_ten.click()
    
def click_difficult_texts(wait_time=0.75):
    sleep(wait_time)
    button_difficult_texts = driver.find_element_by_id('generateDifficultTextsButton')
    button_difficult_texts.click()  
    
def scroll_label_ten(radio_button_id, scroll_wait_seconds = 1.75):
    #we scroll down the results list 
    for scr in range(2,10,2):
        scr_xpath = '//*[@id="group1Table"]/tbody/tr[' + str(scr) + ']/td[1]/a'
        print(scr_xpath)
        link_scroll = driver.find_element_by_xpath(scr_xpath)
        driver.execute_script("return arguments[0].scrollIntoView(true);", link_scroll)
        sleep(scroll_wait_seconds)
        
    radio_buttons = get_radio_buttons()
    radio_buttons[radio_button_id].click()
    sleep(wait_time)
    
    # we apply a group label after checking all 10 suggested are correct
    button_label_ten = driver.find_element_by_id('group1Button')
    sleep(wait_time)
    button_label_ten.click()
    
def search_phrase(phrase):
    # Search for a phrase
    #phrase = "richter"
    element = driver.find_element_by_id("searchAllTexts")
    element.send_keys(phrase)
    driver.find_element_by_id("searchAllTextsButton").click()
    
def search_label(phrases, reps, label_type):
    for r in range(reps):
        for k, v in phrases.items():
            search_phrase(k)
            if label_type=='single':
                select_label_one_text('//*[@id="allTextsTable"]/tbody/tr[' + str (r+1) + ']/td[1]', v, wait_time=wait_time)
        
def get_total_unlabeled(get_labeled=False):
    total_unlabeled = driver.find_element_by_xpath('//*[@id="summarySectionTable"]/tbody/tr[3]/td[2]').text.replace(',', '')
    total_unlabeled = int(total_unlabeled)
    if get_labeled:
        total_labeled = driver.find_element_by_xpath('//*[@id="summarySectionTable"]/tbody/tr[4]/td[2]').text.replace(',', '')
        total_labeled = int(total_labeled)
        return total_unlabeled, total_labeled
    else:
        return total_unlabeled

def get_overall_quality_score():
    overall_quality_score = driver.find_element_by_xpath('//*[@id="difficultTextSummaryTable"]/tbody/tr[1]/td[2]').text
    return overall_quality_score

def export_model():
    driver.find_element_by_id("exportRecordsButton").click()
    # Create a ZipFile Object and load sample.zip in it
    for n in range(5000):
        try:            
            with ZipFile('models/rapid-labeling-results.zip', 'r') as zipObj:
               # Extract all the contents of zip file in current directory
               zipObj.extractall()
            break
        except:
            sleep(0.01)
    os.remove("models/rapid-labeling-results.zip")
    
#def check_if_vectorizer_needs_transform():
    
        
def get_accuracy_score(vectorizer_needs_transform):
    # load the model from disk
    model_filename = os.path.join("output", "trained-classifier.sav")
    loaded_model = pickle.load(open(model_filename, 'rb'))
    if vectorizer_needs_transform:
        #vectorizer_needs_transform = False
        vectorizer_filename = os.path.join("output", "fitted-vectorizer.sav")
        #print(vectorizer_filename)
        vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
        #X_train = vectorizer.transform(train_df["tweet_text"])
        X_test = vectorizer.transform(test_df["tweet_text"])
    y_test = test_df["event_type"]
    y_pred = [x.lower() for x in loaded_model.predict(X_test)]
    test_accuracy_score = accuracy_score(y_test, y_pred)
    #print(test_accuracy_score)
    
    return test_accuracy_score, vectorizer_needs_transform
    
def get_tracker_row(vectorizer_needs_transform, fully_human_labeled=False):
    overall_quality_score = get_overall_quality_score()
    _, total_labeled = get_total_unlabeled(get_labeled=True)
    
    test_accuracy_score = 0.
    
    try:
        export_model()
        test_accuracy_score, vectorizer_needs_transform = get_accuracy_score(vectorizer_needs_transform)
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
    #print(tracker_row)

    return tracker_row, vectorizer_needs_transform

def get_label_id(label):
    radio_button_id = "other"
    if label==str.lower("hurricane"):
        radio_button_id = 0
    if label==str.lower("fire"):
        radio_button_id = 1
    if label==str.lower("flood"):
        radio_button_id = 2
    if label==str.lower("earthquake"):
        radio_button_id = 3
    return radio_button_id

def get_true_label():
    #to mimic full single label at a time human labeling
    tweet_text = driver.find_element_by_xpath('//*[@id="currentTextBody"]/p').text
    tweet_text = str.lower(tweet_text)
    #print(tweet_text)
    true_label = train_df.loc[train_df['tweet_text_lower'].str.contains(tweet_text), "event_type"].values[0]
    
    return true_label

def get_true_label_id():
    true_label = get_true_label()
    true_label_id = get_label_id(true_label)
    return true_label_id


def process_true_labeling(true_labels, df_tracker, vectorizer_needs_transform):
    true_label_id = get_true_label_id() # to mimic full single label at a time human labeling
    #print(true_label_id)
    select_label_one_text(true_label_id, wait_time=0.)
    true_labels += 1
    label_applied = True
    if label_applied==True:
        click_difficult_texts()        
    tracker_row, vectorizer_needs_transform = get_tracker_row(vectorizer_needs_transform, fully_human_labeled=True)
    print(tracker_row)
    df_tracker = df_tracker.append(tracker_row, ignore_index=True)
    
    return true_labels, df_tracker
    

#%%
# Set up accuracy tracker
df_tracker = pd.DataFrame(columns=['labels', 'overall_quality_score', 'accuracy', 'elapsed_time', 'fully_human_labeled'])

# PARAMETERS
initial_true_label_pages = 3
initial_true_label_labels_per_page = 40
true_labeling_cutoff = 37

label_type = "SimilarTexts" # list of valid values ["SimilarTexts", "RecommendedTexts"]
min_recommender_labels = 25 # 3000
max_display_options = 4 # 1-10 at a time, 2-20, 3-50, 4-100, 5-1000 after initial slower single and 10 at a time startup labels
txts_per_page = 15 #50
pages_per_max_display_option = 4 #1071

difficult_texts_per_page = 0 # need to add functionality for this per page on auto-labeling (& manual labeling per page)

label_applied = False

#%%
# Label difficult texts with single labels
# read the contents of the text
sectionstarttime = datetime.datetime.now()

true_labels = 0

for pg in range(initial_true_label_pages):
    if get_total_unlabeled()==0:
        break       
    # loop through page
    for rrow in range(1,initial_true_label_labels_per_page+difficult_texts_per_page+1):
        if get_total_unlabeled()==0:
            break  
        if rrow <= initial_true_label_pages:
            select_tweet_xpath = '//*[@id="allTextsTable"]/tbody/tr[' + str(rrow) + ']/td[1]'
        else:
            select_tweet_xpath = '//*[@id="difficultTextsTable"]/tbody/tr[' + str(rrow-initial_true_label_pages) + ']/td[1]'            
        driver.find_element_by_xpath(select_tweet_xpath).click()
        # label based on text contents
        if get_total_unlabeled()==0:
            break   
        try:
            true_labels, df_tracker = process_true_labeling(true_labels, df_tracker, vectorizer_needs_transform)
        except:
            print("tweet not found")
            pass
                
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

#%%
auto_start_page = int(driver.find_element_by_xpath('//*[@id="allTextTableNextButtons"]/a[3]').text)
print(auto_start_page)

#%%
# read the contents of the text
sectionstarttime = datetime.datetime.now()
phrases = {
    'earthquake': 0,
    'wildfire': 1,
    'hurricane': 3,
    'flooding': 2,
    'fire': 1,
    'richter': 0,
    'smoke': 1,
    'floods': 2,
    'mph': 3,
    'cyclone': 3,
    'heat': 1,
    'quake': 0,
    'tornado': 3,
    'Dorian': 3,
    }

#

for op in range(max_display_options, max_display_options + 1):
    # check how many are unlabelled
    if get_total_unlabeled()==0:
        break   
    if label_type == "SimilarTexts":
        op_base_xpath = '//*[@id="group1_table_limit"]/option['
    elif label_type == "RecommendedTexts":
        op_base_xpath = '//*[@id="group2_table_limit"]/option['
        
    for pg in range(auto_start_page, pages_per_max_display_option + auto_start_page):
        # check how many are unlabelled
        if get_total_unlabeled()==0:
            break        
        # loop through page
        for rrow in range(1,txts_per_page + 1):
            # check how many are unlabelled
            if get_total_unlabeled()==0:
                break
            
            xpath_base = '//*[@id="allTextsTable"]/tbody/tr[' + str(rrow) + ']/td['
            tweet_text = str.lower(driver.find_element_by_xpath(xpath_base + '2]').text)
            #print(tweet_text)
            for k, v in phrases.items():
                # label based on text contents
                if k in tweet_text:
                    # check how many are unlabelled
                    if get_total_unlabeled()==0:
                        break
                    try:
                        select_label_multi_text(xpath_base + '1]/a', v, wait_time=wait_time, max_options=max_display_options, 
                                                label_type=label_type, min_recommender_labels=min_recommender_labels)
                        label_applied = True
                        if label_applied==True:
                            click_difficult_texts()
                        tracker_row, vectorizer_needs_transform = get_tracker_row(vectorizer_needs_transform)
                        print(tracker_row)
                        df_tracker = df_tracker.append(tracker_row, ignore_index=True)
                    except:
                        break
                    break
        
        # go to next page
        driver.find_element_by_xpath('//*[@id="allTextTableNextButtons"]/a[6]').click()    

    # max_display_options = max_display_options - 1

sectionendtime = datetime.datetime.now()
elapsedsectiontime = sectionendtime - sectionstarttime 
print("Elapsed section time", elapsedsectiontime)


#%%
df_tracker.to_csv("tracker_output.csv")
print(df_tracker.head(20))
print(df_tracker.tail(20))

#%%
driver.close()

#%%
#os.remove("output/fitted-vectorizer.sav")

#%%
endtime = datetime.datetime.now()
elapsedtime = endtime - starttime 
print("Elapsed time", elapsedtime)
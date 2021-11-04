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
import numpy as np
import pandas as pd
import re
import string
import os
from time import sleep
import datetime
import pickle

#%%
#set a timer
starttime = datetime.datetime.now()


#%%
# PARAMETERS
mpath = os.getcwd() + "\chromedriver.exe"
wait_time = 0#.75 #0.75
scroll_wait_seconds = 0#1.75 #1.75

#%%
driver = webdriver.Chrome(mpath)

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

def select_label_one_text(xpath, radio_button_id, wait_time=0.75):
    # select a text from the list of all texts
    driver.find_element_by_xpath(xpath).click() 
    sleep(wait_time)
    # we select the correct radio button    
    radio_buttons = get_radio_buttons()
    sleep(wait_time)
    radio_buttons[radio_button_id].click() 
    # label one example
    button_label_single = driver.find_element_by_id('labelButtonSingle')
    button_label_single.click()
    sleep(wait_time)
    
def select_label_multi_text(xpath, radio_button_id, wait_time=0.75, max_options=1):
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
    driver.find_element_by_id("buttonSimilarTexts1Buttons").click()
    # label multi example
    button_label_ten = driver.find_element_by_id('group1Button')
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
    phrase = "richter"
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

def get_tracker_row():
    overall_quality_score = get_overall_quality_score()
    _, total_labeled = get_total_unlabeled(get_labeled=True)
    
    tracker_row = {'labels': total_labeled,
                  'overall_quality_score': overall_quality_score,
                  'accuracy': 0.,
                  }

    return tracker_row

#%%
# #wrap an action that we can repeat
# sectionstarttime = datetime.datetime.now()

# phrases = {
#     'richter': 0,
#     'smoke': 1,
#     'flooding': 2,
#     'mph': 3,
#     }
# reps = 2
# label_type = 'single'

# search_label(phrases, reps, label_type)

# sectionendtime = datetime.datetime.now()
# elapsedsectiontime = sectionendtime - sectionstarttime 
# print("section time", elapsedsectiontime)

#%%
#phrases = {
#    'magnitude': 0,
#    'heat': 1,
#    'floods': 2,
#    'cyclone': 3,
#    }

#search_label(phrases, reps, label_type)
#%%
df_tracker = pd.DataFrame(columns=['labels', 'overall_quality_score', 'accuracy'])
# sample_row = {'labels': 0,
#               'overall_quality_score': 0.,
#               'accuracy': 0.,
#               }
# df_tracker = df_tracker.append(sample_row, ignore_index=True)
# print(df_tracker)

#%%
#tracker_row = get_tracker_row()
#print(tracker_row)

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


max_display_options = 4 # range 1 to 6
txts_per_page = 50
pages_per_max_display_option = 1071 #20

label_applied = False

for op in range(max_display_options + 1):
    # check how many are unlabelled
    if get_total_unlabeled()==0:
        break    
    op_base_xpath = '//*[@id="group1_table_limit"]/option['
    for pg in range(pages_per_max_display_option):
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
                        select_label_multi_text(xpath_base + '1]/a', v, wait_time=wait_time, max_options=max_display_options)
                        label_applied = True
                        if label_applied==True:
                            click_difficult_texts()
                        tracker_row = get_tracker_row()
                        df_tracker = df_tracker.append(tracker_row, ignore_index=True)
                    except:
                        break
                    break
        
        # go to next page
        driver.find_element_by_xpath('//*[@id="allTextTableNextButtons"]/a[6]').click()    

    max_display_options = max_display_options - 1

sectionendtime = datetime.datetime.now()
elapsedsectiontime = sectionendtime - sectionstarttime 
print("Elapsed section time", elapsedsectiontime)


#%%
# # Label difficult texts with single labels
# # read the contents of the text
# sectionstarttime = datetime.datetime.now()
# phrases = {
#     'earthquake': 0,
#     'wildfire': 1,
#     'hurricane': 3,
#     'flooding': 2,
#     'fire': 1,
#     'richter': 0,
#     'smoke': 1,
#     'floods': 2,
#     'mph': 3,
#     'cyclone': 3,
#     'heat': 1,
#     'quake': 0,
#     }

# for pg in range(20):
#     if get_total_unlabeled()==0:
#         break       
#     # loop through page
#     for rrow in range(1,51):
#         if get_total_unlabeled()==0:
#             break       
#         xpath_base = '//*[@id="allTextsTable"]/tbody/tr[' + str(rrow) + ']/td['
#         tweet_text = str.lower(driver.find_element_by_xpath(xpath_base + '2]').text)
#         #print(tweet_text)
#         for k, v in phrases.items():
#             # label based on text contents
#             if get_total_unlabeled()==0:
#                 break       
#             if k in tweet_text:
#                 try:
#                     select_label_one_text(xpath_base + '1]/a', v, wait_time=wait_time)
#                 except:
#                     break
#                 break
    
#     # go to next page
#     driver.find_element_by_xpath('//*[@id="allTextTableNextButtons"]/a[6]').click()    

# sectionendtime = datetime.datetime.now()
# elapsedsectiontime = sectionendtime - sectionstarttime 
# print("Elapsed section time", elapsedsectiontime)
#%%
df_tracker.to_csv("tracker_output.csv")
print(df_tracker.head(20))
print(df_tracker.tail(20))


#%%
endtime = datetime.datetime.now()
elapsedtime = endtime - starttime 
print("Elapsed time", elapsedtime)
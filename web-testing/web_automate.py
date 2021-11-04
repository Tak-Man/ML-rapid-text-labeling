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
import os
from time import sleep
import datetime

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
    # we select the correct radio button    
    radio_buttons = get_radio_buttons()
    sleep(wait_time)
    radio_buttons[radio_button_id].click() 
    # label one example
    button_label_single = driver.find_element_by_id('labelButtonSingle')
    sleep(wait_time)
    button_label_single.click()
    
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
        

#%%
click_difficult_texts(wait_time=wait_time) # this should generate an error message under the labels

#%%
#we select a text Hurricane Dorian
driver.find_element_by_xpath('//*[@id="allTextsTable"]/tbody/tr[2]/td[1]/a').click()

#%%
# we select the Hurricane radio button
radio_buttons = get_radio_buttons()
sleep(wait_time)
radio_buttons[3].click() 

#%%
# we label our first example
button_label_single = driver.find_element_by_id('labelButtonSingle')
sleep(wait_time)
button_label_single.click()

#%%
#we scroll down the results list 
for scr in range(7,27,5):
    scr_xpath = '//*[@id="allTextsTable"]/tbody/tr[' + str(scr) + ']/td[1]/a'
    link_scroll = driver.find_element_by_xpath(scr_xpath)
    driver.execute_script("return arguments[0].scrollIntoView(true);", link_scroll)
    sleep(scroll_wait_seconds)

#%%
#we select a text about floods
driver.find_element_by_xpath('//*[@id="allTextsTable"]/tbody/tr[26]/td[1]/a').click() 

#%%
# we select the flood radio button
radio_buttons = get_radio_buttons()
radio_buttons[2].click()
sleep(wait_time)

#%%
# we label our next example
button_label_single = driver.find_element_by_id('labelButtonSingle')
button_label_single.click()
sleep(wait_time)

#%%
# select another example - this time earthquake
driver.find_element_by_xpath('//*[@id="allTextsTable"]/tbody/tr[1]/td[1]/a').click() 

#%%
# we select the flood radio button
radio_buttons = get_radio_buttons()
radio_buttons[0].click()
sleep(wait_time)

#%%
# we label our next example
button_label_single = driver.find_element_by_id('labelButtonSingle')
button_label_single.click()
sleep(wait_time)

#%%
# use the wrapper function to combine 3 steps needed to select 1. a text, 2. category then 3. apply label
select_label_one_text('//*[@id="allTextsTable"]/tbody/tr[1]/td[1]/a', 1, wait_time=wait_time)

# Label some more examples
#%%
select_label_one_text('//*[@id="allTextsTable"]/tbody/tr[4]/td[1]/a', 2, wait_time=wait_time)

#%%
select_label_one_text('//*[@id="allTextsTable"]/tbody/tr[1]/td[1]/a', 3, wait_time=wait_time)
select_label_one_text('//*[@id="allTextsTable"]/tbody/tr[5]/td[1]/a', 1, wait_time=wait_time)
select_label_one_text('//*[@id="allTextsTable"]/tbody/tr[2]/td[1]/a', 2, wait_time=wait_time)


#%%
select_label_one_text('//*[@id="allTextsTable"]/tbody/tr[2]/td[1]/a', 3, wait_time=wait_time)


#%%
driver.find_element_by_xpath('//*[@id="allTextsTable"]/tbody/tr[2]/td[1]/a').click()

#%%
#we scroll down the results list 
for scr in range(2,10,2):
    scr_xpath = '//*[@id="group1Table"]/tbody/tr[' + str(scr) + ']/td[1]/a'
    link_scroll = driver.find_element_by_xpath(scr_xpath)
    driver.execute_script("return arguments[0].scrollIntoView(true);", link_scroll)
    sleep(scroll_wait_seconds)
    
#%%
# we select the flood radio button
radio_buttons = get_radio_buttons()
radio_buttons[2].click()
sleep(wait_time)
   
#%%
# we apply a group label after checking all 10 suggested are correct
button_label_ten = driver.find_element_by_id('group1Button')
sleep(wait_time)
button_label_ten.click()

#%%
# generate difficult texts
click_difficult_texts(wait_time=wait_time)

#%%
# select a text from the list of all texts
driver.find_element_by_xpath('//*[@id="difficultTextsTable"]/tbody/tr[8]/td[1]/a').click() 

#%%
# Check 10 then group label 10
scroll_label_ten(0, scroll_wait_seconds=scroll_wait_seconds)

#%%
#wrap an action that we can repeat
sectionstarttime = datetime.datetime.now()

phrases = {
    'richter': 0,
    'smoke': 1,
    'flooding': 2,
    'mph': 3,
    }
reps = 2
label_type = 'single'

search_label(phrases, reps, label_type)

sectionendtime = datetime.datetime.now()
elapsedsectiontime = sectionendtime - sectionstarttime 
print("section time", elapsedsectiontime)

#%%
#phrases = {
#    'magnitude': 0,
#    'heat': 1,
#    'floods': 2,
#    'cyclone': 3,
#    }

#search_label(phrases, reps, label_type)
#%%
# read the contents of the text
sectionstarttime = datetime.datetime.now()
phrases = {
    'earthquake': 0,
    'wildfire': 1,
    'flooding': 2,
    'hurricane': 3,
    'richter': 0,
    'smoke': 1,
    'floods': 2,
    'mph': 3,
    'cyclone': 3,
    'heat': 1,
    'quake': 0,
    }
for rrow in range(1,51):
    xpath_base = '//*[@id="difficultTextsTable"]/tbody/tr[' + str(rrow) + ']/td['
    
    tweet_text = driver.find_element_by_xpath(xpath_base + '2]').text
    for k, v in phrases.items():
        print(rrow, k)
        if k in str.lower(tweet_text):
            select_label_one_text(xpath_base + '1]/a', v, wait_time=wait_time)
            break
    
# label based on text contents

sectionendtime = datetime.datetime.now()
elapsedsectiontime = sectionendtime - sectionstarttime 


#%%
endtime = datetime.datetime.now()
elapsedtime = endtime - starttime 
print("Elapsed time", elapsedtime)
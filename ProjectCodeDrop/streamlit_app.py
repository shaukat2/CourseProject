#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import urllib
import requests
import time
import pandas as pd
import numpy as np
from numpy import random


# In[2]:


user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent,} 

STOPWORDS = set(stopwords.words('english'))
ProcessedText = None

def GetDataFromURL(url_link):
    time.sleep(0.01)
    response = requests.get(url_link)
    soup = BeautifulSoup(response.text, "html.parser")
    remove_script(soup)
    text = soup.get_text()
    preprocessed_text = text
    #preprocessed_text = preprocess_text(text)
    return preprocessed_text

#Checks if bio_url is a valid faculty homepage
def is_valid_url(url_check):
    ret_url = 'NA'
    if url_check == 'NA':
        return ret_url
    if url_check.endswith('.pdf'): #we're not parsing pdfs
        return ret_url
    try:
        #sometimes the homepage url points to the same page as the faculty profile page
        #which should be treated differently from an actual homepage
        request=urllib.request.Request(url_check,None,headers)
        ret_url = urllib.request.urlopen(request).geturl() 
    except:
        return ret_url      #unable to access bio_url
    return ret_url


def remove_script(soup):
    for script in soup(["script", "style"]):
        script.decompose()
    return soup

def scrapeURL(url_in):
    url_check = is_valid_url(url_in)
    if url_check != 'NA':
        ProcessedText = GetDataFromURL(url_check)
    else:
        ProcessedText = "NA"
    return ProcessedText



def preprocess_text(ExtractedText):
    ExtractedText = " ".join((re.sub(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+', "EmailAddress", ExtractedText)).split())
    ExtractedText = " ".join((re.sub(r'^https?:\/\/.*[\r\n]*',"WebAddress", ExtractedText)).split())
    ExtractedText = ExtractedText.encode('ascii',errors='ignore').decode('utf-8')       #removes non-ascii characters
    ExtractedText = re.sub('\s+',' ',ExtractedText)       #repalces repeated whitespace characters with single space
    ExtractedText = re.sub(r'\W',' ',ExtractedText) 
    ExtractedText = ExtractedText.replace("\n"," ")
    ExtractedText = ExtractedText.lower()
    ExtractedText = ' '.join(word for word in ExtractedText.split() if word not in STOPWORDS) # delete stopwors from text
    return ExtractedText


clf_rptFacDir = pd.read_csv("clf_rptFacDir.csv",index_col=[0])
clf_rptFac = pd.read_csv("clf_rptFac.csv",index_col=[0])
# In[3]:


import pickle
with open('Fac_classifier', 'rb') as training_model:
    FacClass = pickle.load(training_model)
with open('FacDir_classifier', 'rb') as training_model:
    FacDirClass = pickle.load(training_model)



import streamlit as st

st.title("Faculty / Non-Faculty Classification")

st.markdown("""
<style>
.big-font {
    font-size:16px !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">Enter URL in textbox in left side-bar to see classifier results</p>', unsafe_allow_html=True)

st.markdown('<p class="big-font">Classification Report for Faculty Directory/Non-Faculty Classifier</p>', unsafe_allow_html=True)
st.write(clf_rptFacDir)

st.markdown('<p class="big-font">Classification Report for Faculty/Non-Faculty Classifier</p>', unsafe_allow_html=True)
st.write(clf_rptFac)

st.markdown('<p class="big-font">What I tried</p>', unsafe_allow_html=True)
st.text("1. Manual Feature Extraction but used automatic feature extraction.")
st.text("2. Multiple Classification Model Libraries, chose SGD from sklearn for this project.")

st.markdown('<p class="big-font">Future Improvements</p>', unsafe_allow_html=True)
st.text("1. Multi-Class Calssification can be added to improve utility.")
st.text("2. Can be added ExpertSearchSystem to improve utility.")

user_input = st.sidebar.text_area("Enter URL to check if Faculty/Non-Faculty Page", "https://www.google.com", key="1")
text = user_input
text = is_valid_url(user_input)
if text == 'NA':
    rep = "Your input is invalid."
else:
    ProcessedText = scrapeURL(user_input)
    ProcessedText = preprocess_text(ProcessedText)
    rep = FacClass.predict([ProcessedText])
    rep = rep[0]

st.sidebar.write({rep})

    
user_input = st.sidebar.text_area("Enter URL to check if Faculty Directory Page/Non-Faculty Page", "https://www.google.com", key="2")
text = user_input
text = is_valid_url(user_input)
if text == 'NA':
    rep = "Your input is invalid."
else:
    ProcessedText = scrapeURL(user_input)
    ProcessedText = preprocess_text(ProcessedText)
    rep = FacDirClass.predict([ProcessedText])
    rep = rep[0]

st.sidebar.write({rep})
        
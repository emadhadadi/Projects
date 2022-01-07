# -*- coding: utf-8 -*-


#                  _______________________________________________________
#                  | Import all libraries that will be used in the project |

import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('arabic')
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from pyarabic.araby import strip_tashkeel 
from pyarabic.araby import strip_tatweel 
from pyarabic.araby import normalize_hamza
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, plot_confusion_matrix, precision_score, recall_score


# The Arabic Hate speech Lixcon  
#___________________________________
HS = pd.read_excel('Lexicon.xlsx' ) #|
HS = HS.squeeze()                  #|
HS = list(HS)                      #|
#__________________________________#|


#   __________________________
#   |Loading Training Dataset |
#___________________________________________________________________________________
train = pd.read_excel('Hate speech 20 nov.xlsx' , sheet_name ="new_Training_data") #|
combi = train.copy()                                                               #|
#__________________________________________________________________________________#|

#                         _________________________________________
#                         | Features extraction and data cleaning |
#
# This function extracts the following properties (the number of stop words, 
# the length of each tweet, and the number of words that match the Lexicon)
#  and then cleans and prepares the data for use in the model

def pre_processing(combi):
    combi["HS"] = combi['text'].apply(lambda x: len([x for x in x.split() if x in HS]))
    combi['word_count']= combi['text'].apply(lambda x : len(str(x).split(' ')))
    combi['stopwords_count'] = combi['text'].apply(lambda x: len([x for x in x.split() if x in stop]))
    combi['text'] = combi['text'].apply(lambda x: str(strip_tashkeel(x)))
    combi['text'] = combi['text'].apply(lambda x: str(strip_tatweel(x)))
    combi['text'] = combi['text'].str.replace("[إأٱآا]", "ا")
    combi['text'] = combi['text'].str.replace("[ؤ]", "و")
    combi['text'] = combi['text'].str.replace('[\d]',' ')
    combi['text'] = combi['text'].str.replace('[^\w\s]',' ')
    combi['text'] = combi['text'].str.replace('[a-zA-Z]',' ') 
    combi['text'] = combi['text'].str.replace('[_]',' ')
    combi['text'] = combi['text'].apply(lambda x: " ".join(x.split()))
    combi['text'] = combi['text'].str.strip()
    combi['text'].replace([" ",""] , np.nan, inplace=True)
    combi.dropna(subset=['text'], inplace=True)
    combi['text'] = combi['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    st = ISRIStemmer()
    combi['text'] = combi['text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()])) 
    return combi

    


combi = pre_processing(combi)


#___________________________________________________________________________________________________



#                  ______________________________________________
#                 | Term Frequency–Inverse Document Frequency    |

# We tested the bag of word(BOW) and tf-idf and chose tf-idf because it gives better results
tfidf_vectorizer = TfidfVectorizer(max_df=0.8,min_df=5 ) # ngram_range=(1,2)
tfidf = tfidf_vectorizer.fit_transform(combi['text'])


Basic_Featue_Extraction = combi[[ 'HS' , 'word_count', 'stopwords_count']].to_numpy()
Basic_Featue_Extraction = Basic_Featue_Extraction.astype(float)

append = np.concatenate((tfidf.toarray(), Basic_Featue_Extraction), axis=1)





#                  ______________________________________________
#                 |                 Model Building               |

# we chose the random forest algorithm after trying many machine learning algorithms

#After Basic_Featue_Extraction
train_tf = append[0:]

# splitting data into training and validation set
Tr_D_bow, Te_D_bow, Tr_L_bow , Te_L_bow = train_test_split(train_tf, combi['class'], random_state=45, test_size=0.2)
# Create Random Forest classifer object
rm = RandomForestClassifier()
# Train Decision Tree Classifer
rm = rm.fit(Tr_D_bow,Tr_L_bow)
#Predict the response for test dataset
rm_pre = rm.predict(Te_D_bow)

# The highest accuracy we got is 92% .
print('The Accuracy of RF is -->',metrics.accuracy_score(Te_L_bow, rm_pre))
print(classification_report(Te_L_bow,rm_pre))

print(confusion_matrix(Te_L_bow, rm_pre)) 

# Calculate accuracy manually
(581+520)/(581+520+33+57)

#                     _____________________________________________________
#____________________| Model Evaluation & Testing  using out of sample Data| ____________

train1 = pd.read_excel('Testing.xlsx', sheet_name="testing 2")

combi1 = train1.copy()

combi1['Tweet'] = combi1['Tweet'].astype(str)


def pre_processing1(combi1):
    combi1["HS"] = combi1['Tweet'].apply(lambda x: len([x for x in x.split() if x in HS]))
    combi1['word_count']= combi1['Tweet'].apply(lambda x : len(str(x).split(' ')))
    combi1['stopwords_count'] = combi1['Tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
    combi1['Tweet'] = combi1['Tweet'].apply(lambda x: str(strip_tashkeel(x)))
    combi1['Tweet'] = combi1['Tweet'].apply(lambda x: str(strip_tatweel(x)))
    combi1['Tweet'] = combi1['Tweet'].str.replace("[إأٱآا]", "ا")
    combi1['Tweet'] = combi1['Tweet'].str.replace("[ؤ]", "و")
    combi1['Tweet'] = combi1['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    combi1['Tweet'] = combi1['Tweet'].str.replace('[\d]',' ')
    combi1['Tweet'] = combi1['Tweet'].str.replace('[^\w\s]',' ')
    combi1['Tweet'] = combi1['Tweet'].str.replace('[a-zA-Z]',' ') 
    combi1['Tweet'] = combi1['Tweet'].str.replace('[_]',' ')  
    combi1['Tweet'] = combi1['Tweet'].apply(lambda x: " ".join(x.split()))
    combi1['Tweet'] = combi1['Tweet'].str.strip()
    combi1['Tweet'].replace([" ",""] , np.nan, inplace=True)
    combi1.dropna(subset=['Tweet'], inplace=True)
    combi1['Tweet'] = combi1['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    st = ISRIStemmer()
    combi1['Tweet'] = combi1['Tweet'].apply(lambda x: " ".join([st.stem(word) for word in x.split()])) 
    return combi1

combi1 = pre_processing1(combi1)


tfidf1 = tfidf_vectorizer.transform(combi1['Tweet'])

Basic_Featue_Extraction1 = combi1[[ 'HS' , 'word_count', 'stopwords_count']].to_numpy()
Basic_Featue_Extraction1 = Basic_Featue_Extraction1.astype(float)

append1 = np.concatenate((tfidf1.toarray(), Basic_Featue_Extraction1), axis=1)

predict = rm.predict( append1[0:])
predict = pd.DataFrame(predict)
combi1['Class'] = predict
# To reveiw results in variable explorer in pre-cleaning part tweet with predicted column
train1['Class'] = predict

print('The Accuracy of RF is -->',metrics.accuracy_score(train1['actual'], train1['Class']))



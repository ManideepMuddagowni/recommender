from flask import Flask,render_template,url_for,request
import pickle
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer #Import TfIdfVectorizer from scikit-learn
from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize import RegexpTokenizer
import re
import string
import random
import requests
from io import BytesIO
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from Remove_Punctuations import remove_punctuation
from _RemoveNonAscii import _removeNonAscii
from Clean_Data import clean_data
from Get_Recommendations import get_recommendations
from Rec_Lin import rec_lin
from Make_lower_case import make_lower_case
from Remove_html import remove_html
from TF_IDF_Recommender import TF_IDF_recommender
from Get_Recommendation import get_recommendation
from CVextor_Recommender import CVextor_recommender
from KNN_Recommender import KNN_recommender

app = Flask(__name__)
myshoplivery_data = pd.read_csv('export_catalog_product_20221219_071525 (1).csv', low_memory=False)

myshoplivery_data['description'] = myshoplivery_data['description'].dropna(axis=0)

features = ['description','categories']
for feature in features:
    myshoplivery_data[feature] = myshoplivery_data[feature].apply(clean_data)

myshoplivery_data['description'] = myshoplivery_data['description'].apply(_removeNonAscii)
myshoplivery_data['description'] = myshoplivery_data['description'].apply(func = make_lower_case)
myshoplivery_data['description'] = myshoplivery_data['description'].apply(func=remove_punctuation)
myshoplivery_data['description'] = myshoplivery_data['description'].apply(func=remove_html)

myshoplivery_data['categories'] = myshoplivery_data['categories'].apply(_removeNonAscii)
myshoplivery_data['categories'] = myshoplivery_data['categories'].apply(func = make_lower_case)
myshoplivery_data['categories'] = myshoplivery_data['categories'].apply(func=remove_punctuation)
myshoplivery_data['categories'] = myshoplivery_data['categories'].apply(func=remove_html)

myshoplivery_data.dropna()

tf = TfidfVectorizer(ngram_range=(2, 2), stop_words='english', lowercase = False)
tfidf_matrix = tf.fit_transform(myshoplivery_data['description'])
total_words = tfidf_matrix.sum(axis=0) 

freq = [(word, total_words[0, idx]) for word, idx in tf.vocabulary_.items()]
freq =sorted(freq, key = lambda x: x[1], reverse=True)

myshoplivery_data['item_name'] = myshoplivery_data['name']

myshoplivery_data["text"]=  myshoplivery_data['description'] + ' '+ myshoplivery_data['item_name'] + ' ' + myshoplivery_data['categories'] 
df_shop= myshoplivery_data[['sku','name','text','price']]

df_shop['name'] = df_shop['name'].apply(_removeNonAscii)
df_shop['name'] = df_shop['name'].apply(func = make_lower_case)
df_shop['name'] = df_shop['name'].apply(func=remove_punctuation)
df_shop['name'] = df_shop['name'].apply(func=remove_html)

df_shop=df_shop[df_shop['sku']!='Test']

df_shop['sku']=df_shop['sku'].apply(lambda x: int((x)))

tf = TfidfVectorizer(ngram_range=(2, 2), stop_words='english', lowercase = False)
tfidf_matrix = tf.fit_transform(df_shop['text'])
total_words = tfidf_matrix.sum(axis=0) 

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df_shop['text'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df_shop.index, index=df_shop['name']).drop_duplicates()

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df_shop['text'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df_shop = df_shop.reset_index()
indices1 = pd.Series(df_shop.index, index=df_shop['name'])

@app.route('/')
def home():
	return render_template('input.html')

#p=input('Enter item name:')
#price_choice=float(input('Enter Price of Item:'))
#p=_removeNonAscii(p)
#p =  make_lower_case(p)
#p = remove_punctuation(p)
#p = remove_html(p)
#print('Without Optimization')
#print(get_recommendations(p,cosine_sim2,cosine_sim,indices,df_shop))''''
@app.route('/search_product',methods=['GET','POST'])
def search_product():
    #m = int(input('Enter sku : '))# 135 #the user 2(using index)
    return render_template('input.html')
    
@app.route('/result', methods=['GET','POST'])
def result():
    m=int(request.form['p_s'])
    user= df_shop[['sku','text','price']]
    user_q =user[user['sku']==m]
    u=user[user['sku']==m].index[0]
    #user_q = user.loc[[user[user['sku']==u]]]
    print(user_q)

    from sklearn.feature_extraction.text import TfidfVectorizer

    #print('TF-IDF Reccomendation')
    #print(TF_IDF_recommender(user_q,df_shop,u))

    #print('Count-Vectorizer Reccomendation')
    #print(CVextor_recommender(user_q, df_shop, u))

    #print('KNN Reccomendation')
    #print(KNN_recommender(user_q,df_shop, u ))

    #item_name=["MSI Gaming Chair Comfortable and available in X "]

    #item_series=myshoplivery_data.name.isin(item_name)

    #filtered_myshoplivery_data=myshoplivery_data[item_series]
    #print(filtered_myshoplivery_data)

    p=CVextor_recommender(user_q,df_shop,u)['sku_rec'].map(lambda x: int(x))

    #df_shop.loc[df_shop['sku'].isin(p)]

    p=p.apply(lambda x:int( x))
    df_shop['sku']=df_shop['sku'].apply(lambda x: int(x))
    df_shop.loc[df_shop.apply(lambda x: x['sku'] in p, axis=1)]

    from copy import deepcopy
    df_new1=deepcopy(df_shop)
    df_new1['sku']
    df_new1.set_index('sku', inplace=True)
    df_new1=df_new1.loc[p]


    #price_choice=75000
    price_choice=float(user_q['price'][u])
    #df_new1 = df_new1[['sku',"name","price"]].loc[index_fetched]
    df_new1=df_new1[df_new1['price'] < (price_choice+(0.2*price_choice))]
    df_new1=df_new1[(price_choice-(0.2*price_choice))<df_new1['price']]
    df_new1=df_new1[:16]
    #print(df_new1)
    #print(df_new1.index)
    return render_template('out.html',tables=[df_new1[['index','name','price']].to_html(classes='data')], titles=df_new1[['index','name','price']].columns.values)

if __name__ == '__main__':
	#app.run(debug=True)
    app.run(host='127.0.0.1', port=8001, debug=True)
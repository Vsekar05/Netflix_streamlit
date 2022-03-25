import pandas as pd
import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import re
import string


Data=pd.read_csv("https://raw.githubusercontent.com/Vsekar05/Datasets/main/netflix.csv")
Data["user rating score"] = Data["user rating score"].dropna()
mean_value=np.mean(Data["user rating score"])
Data["user rating score"].fillna(value=mean_value, inplace=True)

le=preprocessing.LabelEncoder()
Data["rating"]=le.fit_transform(Data["rating"])

Data["ratingLevel"].fillna(value="Parents strongly cautioned. May be unsuitable for children ages 14 and under.", inplace=True)

def text_clean_1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

cleaned1 = lambda x: text_clean_1(x)

Data['cleaned_ratinglevel'] = pd.DataFrame(Data.ratingLevel.apply(cleaned1))


Independent_var = Data.cleaned_ratinglevel
Dependent_var = Data.rating

IV_train, IV_test, DV_train, DV_test = train_test_split(Independent_var, Dependent_var, test_size = 0.1, random_state = 225)

tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver = "lbfgs")

model = Pipeline([('vectorizer',tvec),('classifier',clf2)])

model.fit(IV_train, DV_train)

predictions = model.predict(IV_test)

confusion_matrix(predictions, DV_test)

Show_Id=[]
for x in range(1,1001):
  Show_Id.append(x)
Data["Show_Id"]=Show_Id

New_Data=Data.pivot(index="rating",columns="Show_Id",values="user rating score")

New_Data.fillna(0,inplace=True)

csr_data = csr_matrix(New_Data.values)
New_Data.reset_index(inplace=True)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

def get_show_recommendation(show_name):
    n_movies_to_reccomend = 10
    movie_list = Data[Data['title'].str.contains(show_name)]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['rating']
        movie_idx = New_Data[New_Data['rating'] == movie_idx].index[0]
        
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),\
                               key=lambda x: x[1])[:0:-1]
        
        recommend_frame = []
        
        for val in rec_movie_indices:
            movie_idx = New_Data.iloc[val[0]]['rating']
            idx = Data[Data['rating'] == movie_idx].index
            recommend_frame.append({'Title':Data.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    
    else:
        
        return "No movies found. Please check your input"

st.set_page_config(
    page_title="Netflix shows rating analyser and Recomendation",
    page_icon="https://miro.medium.com/max/807/1*uxkCLCVoxLJlkoPTFDoMLg.jpeg",
    layout="centered", initial_sidebar_state="auto"
)
rating_level=[st.text_input(label="Give the rating level",value="",max_chars=200, key=None, type="default",
              placeholder=None, disabled=False)]
Rating=model.predict(rating_level)
st.write('The rating for the given movie is',Rating)
Show_name=st.text_input(label="Give the show title",value="",max_chars=200, key=None, type="default",
              placeholder=None, disabled=False)
Recomended_movies=get_show_recommendation(Show_name)
 
st.write("The shows which are similar in rating as the show which you have given are",Recomended_movies)

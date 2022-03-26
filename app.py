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
st.image("data:image/png;base64,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")
st.title("Netflix show Rating Analyzer and Recommendation")
rating_level=[st.text_input(label="Give the rating level",value="",max_chars=200, key=None, type="default",
              placeholder=None, disabled=False)]
Rating=model.predict(rating_level)
st.write('The rating for the given movie is',Rating)
Show_name=st.text_input(label="Give the show title",value="",max_chars=200, key=None, type="default",
              placeholder=None, disabled=False)
Recomended_movies=get_show_recommendation(Show_name)
 
st.write("The shows which are similar in rating as the show which you have given are",Recomended_movies)

from flask import Flask, request
import random
import joblib
import pandas as pd
from nltk.corpus import stopwords
import nltk

nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer
import re

text_data = pd.read_csv("Emotion_final.csv")
song_data=pd.read_csv("VybeSongs.csv")
for i in range(len(text_data)):
    if(text_data.loc[i, "Emotion"] == "anger"):
                text_data.at[i,"Emotion"]="angry"
    if(text_data.loc[i, "Emotion"] == "sadness"):
                text_data.at[i,"Emotion"]="sad"
numberCol=[]
for i in range(len(text_data)):
    if(text_data.loc[i, "Emotion"] == "happy"):
                numberCol.append(0)
    if(text_data.loc[i, "Emotion"] == "sad"):
                numberCol.append(1)
    if(text_data.loc[i, "Emotion"] == "angry"):
                numberCol.append(2)
    if(text_data.loc[i, "Emotion"] == "love"):
                numberCol.append(3)
    if(text_data.loc[i, "Emotion"] == "surprise"):
                numberCol.append(4)
    if(text_data.loc[i, "Emotion"] == "fear"):
                numberCol.append(5)
text_data["Target"]=numberCol
X=text_data.values[:, 0]
y=text_data.values[:, 2]

documents=[]
for sen in range(0, len(X)):
            document = re.sub(r'\W', ' ', str(X[sen]))
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
            document = re.sub(r'\s+', ' ', document, flags=re.I)
            document = re.sub(r'^b\s+', '', document)
            document = document.lower()
            document = document.split()
            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)
            documents.append(document)
        
tfidfconverter1 = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter1.fit_transform(documents).toarray()
app = Flask(__name__)
file = open('emomodel.pkl', 'rb')
loaded_model = joblib.load(file)
@app.route("/")
@app.route("/predict", methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        data=request.args.get('data')
        #data=data.split(":")
        txt=str(data)
        documents=[]
        document = re.sub(r'\W', ' ', str(txt))
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        document = re.sub(r'^b\s+', '', document)
        document = document.lower()
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)           
        documents.append(document)
        X = tfidfconverter1.transform(documents).toarray()

        [prediction] = loaded_model.predict(X)
        emotion_dict = {
            0: "happy",
            1: "sad",
            2: "angry",
            3: "love",
            4: "surprise",
            5: "fear"
        }
        emotion=emotion_dict[prediction]
        print(emotion)
        relevant_emo=[]
        for i in range(len(song_data)):
            if(song_data.loc[i, "emotion"] == str(emotion)):
                relevant_emo.append(song_data.loc[i,"Track"]+" by "+song_data.loc[i,"Artist"])
        song=random.choice(relevant_emo)
        return song
if __name__ == '__main__':
    app.run(  debug=True)
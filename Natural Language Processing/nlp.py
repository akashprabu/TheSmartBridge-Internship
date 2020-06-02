import numpy as np
import pandas as pd

dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter = '\t')
import re #regular expressions to replace special charecters
import nltk
nltk.download("stopwords") # for is then that or , is here where
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # is used to stem the word
ps = PorterStemmer()
data = []
for i in range(0,1000):
    review = dataset["Review"][i]
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    data.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
x = cv.fit_transform(data).toarray()

y = dataset.iloc[:,1:2].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units = 1565 ,init = "random_uniform",activation = "relu"))
model.add(Dense(units = 3000 ,init = "random_uniform",activation = "relu"))

model.add(Dense(units = 1 ,init = "random_uniform",activation = "sigmoid"))
model.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
model.fit(x_train,y_train,epochs  = 50)
x_train.shape




y_pred = model.predict(x_test)

y_pred = (y_pred >0.5)

y_p = model.predict(cv.transform(["good"]))
y_p = y_p>0.5


text =  "wow...... it was amazing tasty food"
text = re.sub('[^a-zA-Z]', ' ',text)
text = text.lower()
text = text.split()
text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
text = ' '.join(text)

y_p1 = model.predict(cv.transform([text]))
y_p1 = y_p>0.5







    









"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from textwrap import wrap
from textblob import TextBlob
import joblib
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

text=pd.read_csv('AmazonReviewsDataset.csv')
textdata = text[['Rating','Review']]
textdata.dropna(inplace=True)
textdata['Rating']=textdata['Rating'].apply(lambda x: float(x.split(' out ')[0]))
textdata['Review']=textdata['Review'].apply(lambda x: x.lower())
textdata['Review']=textdata['Review'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
stopwords = set(STOPWORDS)       
stopwords.add('router')
textdata['Review'] = textdata['Review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
textdata['polarity'] = textdata['Review'].map(lambda text: TextBlob(text).sentiment.polarity)
textdata['review_len'] = textdata['Review'].astype(str).apply(len)
textdata['word_count'] = textdata['Review'].apply(lambda x: len(str(x).split()))
l=[]
for (i,j) in zip(textdata['polarity'],textdata['Rating']):
  if i>=0.3 and j>=4:
    l.append(1)
  else:
    l.append(0)
textdata['target']=pd.Series(l)
textdata.dropna(inplace=True)
textdata['target']=textdata['target'].astype(int)
l=[]
for i in textdata['Review']:
  l.append(i)
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(l).toarray()
X=pd.DataFrame(X)
X['Rating']=textdata['Rating']
X['target']=textdata['target']
X.dropna(inplace=True)
Y=X.target.values
joblib.dump(cv, "cv.pkl")
X.drop(['Rating','target'],axis = 1, inplace = True)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)
rf=RandomForestClassifier(n_estimators=100,oob_score=True,max_features=5)

rf.fit(x_train,y_train)

joblib.dump(rf, "model.pkl")"""




# Importing libraries
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib


# Loading the dataset
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')


corpus = []

# Looping till 1000 because the number of rows are 1000
for i in range(0, 1000):
    # Removing the special character from the reviews and replacing it with space character
    review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df['Review'][i])

    # Converting the review into lower case character
    review = review.lower()

    # Tokenizing the review by words
    review_words = review.split()

    # Removing the stop words using nltk stopwords
    review_words = [word for word in review_words if not word in set(
        stopwords.words('english'))]

    # Stemming the words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review_words]

    # Joining the stemmed words
    review = ' '.join(review)

    # Creating a corpus
    corpus.append(review)


# Creating Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

# Creating a pickle file for the CountVectorizer model
joblib.dump(cv, "cv.pkl")


# Model Building
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)

# Creating a pickle file for the Multinomial Naive Bayes model
joblib.dump(classifier, "model.pkl")

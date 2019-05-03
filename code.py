import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

data = pd.read_csv('googleplaystore_user_reviews.csv')
data.head()

Data = data.dropna()
Data.head()

Data.Sentiment[Data.Sentiment =='Positive'] = 0
Data.Sentiment[Data.Sentiment =='Neutral'] = 1
Data.Sentiment[Data.Sentiment =='Negative'] = 2
Data.head()

google = Data[Data['App'].str.contains("Chat")] 
google.Sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["green", "red", "yellow"])

google['Sentiment'] = google['Sentiment'].astype('int')

google["index"] = range(0,len(google))
google = google.set_index("index")
google.shape

corpus=[]
for i in range(0, len(google)):  
      
    # column : "Review", row ith 
    review = re.sub('[^a-zA-Z]', ' ', google['Translated_Review'][i])  
      
    # convert all cases to lower cases 
    review = review.lower()  
      
    # split to array(default delimiter is " ") 
    review = review.split()  
      
    # creating PorterStemmer object to 
    # take main stem of each word 
    ps = PorterStemmer()  
      
    # loop for stemming each word 
    # in string array at ith row     
    review = [ps.stem(word) for word in review 
                if not word in set(stopwords.words('english'))]  
                  
    # rejoin all string array elements 
    # to create back into a string 
    review = ' '.join(review)   
      
    # append each string to create 
    # array of clean text  
    corpus.append(review) 
    
    words=[]
for i in range(0,len(corpus)):
    words = words + (re.findall(r'\w+', corpus[i]))# words cantain all the words in the dataset

words = [word for word in words if len(word) > 2]

from collections import Counter
words_counts = Counter(words)
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)
most_commmom_wordList = []
most_commmom_CountList = []
for x, y in most_common_words:
    most_commmom_wordList.append(x)
    most_commmom_CountList.append(y)
    
plt.figure(figsize=(20,18))
plot = sns.barplot(np.arange(30), most_commmom_CountList[0:30]) #width=0.35)
plt.ylabel('Word Count',fontsize=20)
plt.xticks(np.arange(30), most_commmom_wordList[0:30], fontsize=20, rotation=40)
plt.title('Most Common Word used in the Review.', fontsize=20)
plt.show()

k = most_commmom_wordList[0:20]
Sentiment_Polarity=[]
Positive=[]
Neutral=[]
Negative=[]
for i in k:
    Sentiment=[]
    for z in corpus:
        if i in z and google['Sentiment'][corpus.index(z)]==0:
            Positive.append(i)
        if i in z and google['Sentiment'][corpus.index(z)]==1:
            Neutral.append(i)
        if i in z and google['Sentiment'][corpus.index(z)]==2:
            Negative.append(i)
     
f,ax = plt.subplots(3,1,figsize=(15,18))
c1 = sns.countplot(Positive,ax=ax[0])
c2 = sns.countplot(Neutral,ax=ax[1])
c3 = sns.countplot(Negative,ax=ax[2])
ax[0].set_title("Number of times Most Common Words \nused in case of POSITIVE Reviw",fontsize=20)
ax[1].set_title("Number of times Most Common Words \nused in case of NEUTRAL Reviw",fontsize=20)
ax[2].set_title("Number of times Most Common Words \nused in case of NEGATIVE Reviw",fontsize=20)
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
y = google['Sentiment'][:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#lr = LogisticRegression()
#lr = DecisionTreeClassifier()
lr = SVC(kernel='linear')
#lr = KNeighborsClassifier()
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

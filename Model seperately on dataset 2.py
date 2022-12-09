#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the library
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


import nltk
nltk.download('stopwords')


# In[3]:


dataset = pd.read_csv('Desktop/WELFake_Dataset.csv')


# In[4]:


dataset= dataset[0:20000]


# In[5]:


dataset


# # Preprocessing the dataset

# In[6]:


# shape of the dataset

dataset.shape


# In[7]:


# viewing first few rows of the dataset

dataset.head()


# In[8]:


# checking for missing values

dataset.isnull().sum()


# In[9]:


# replacing null values with empty string

dataset = dataset.fillna("")


# In[10]:


dataset


# In[11]:


#combine title and text
dataset['content'] = dataset['title'] + ' ' + dataset['text']


# In[12]:


print(dataset['content'])


# # Stemming

# In[13]:


#stemming is neccessary in order remove the excessive noise in the datasets
port_stem = PorterStemmer()


# In[14]:


# stemming function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[15]:


# applying stemming function
dataset['content'] = dataset['content'].apply(stemming)


# In[16]:


# viewing content after stemming
print(dataset['content'])


# In[26]:


#separating data and label

X = dataset['content'].values
Y= dataset['label'].values


# In[27]:


X


# In[28]:


# converting textual data to numerical data

vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)


# In[29]:


print(X)


# # Single Word Cloud

# In[30]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# initialize the word cloud
wordcloud = WordCloud( background_color='black', width=800, height=600)
# generate the word cloud by passing the news
text_cloud = wordcloud.generate(' '.join(dataset['text']))
# plotting the word cloud
plt.figure(figsize=(20,30))
plt.imshow(text_cloud)
plt.axis('off')
plt.show()


# # Data Splitting to train and test 

# In[22]:


# splitting data to training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[23]:


X_train


# In[24]:


Y_test


# In[25]:


Y_train


# In[26]:


X_test


# # Model training using Logistic regression

# In[27]:


model = LogisticRegression()


# In[28]:


model.fit(X_train, Y_train)


# In[29]:


# accuracy score on training data
X_train_predictions = model.predict(X_train)
accuracy_training_data = accuracy_score(X_train_predictions, Y_train)
print(f"Accuracy score for training data: {accuracy_training_data}")


# In[30]:


# accuracy score on test data
X_test_predictions = model.predict(X_test)
accuracy_test_data = accuracy_score(X_test_predictions, Y_test)
print(f"Accuracy score for test data: {accuracy_test_data}")


# # To build a predictive system for logistic regression

# In[40]:


# Build a Simple Predictive System

'''
index = int(input("Enter article number to be verified: "))
^ To get article number as input from user
'''

X_new = X_test[1]
new_predict = model.predict(X_new)
if(new_predict[0]==0):
    print("The News is fake")
else:
    print("The News is real")


# In[69]:


print(Y_test[1])


# # Classification report for logistic regression

# In[43]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, X_test_predictions))


# # Classification Model: Multinomial Naive Bayes

# In[45]:


#let's implement the model : Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


# In[46]:


#testing and training the models seperately
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[47]:


X1_train


# In[53]:


import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[54]:


from sklearn import metrics
import numpy as np
import itertools

classifier.fit(X1_train, Y1_train)
prediction1 = classifier.predict(X1_test)
score = metrics.accuracy_score(Y1_test, prediction1)
print("accuracy:   %0.3f" % score)


# In[55]:


cm1 = metrics.confusion_matrix(Y1_test, prediction1)
plot_confusion_matrix(cm1, classes=['FAKE', 'REAL'])
plt.show()


# # Predictive system for Multinomial Naive Bayes

# In[62]:


X1_new = X1_test[3000]

prediction = model.predict(X1_new)
print(prediction)

if (prediction[0]==0):
  print('The news is fake')
else:
  print('The news is real')


# In[66]:


print(Y1_test[3000])


# # Classification report for Multinomial Naive Bayes

# In[59]:


from sklearn.metrics import classification_report
print(classification_report(Y1_test,prediction1))


# # Passive Agressive classifier

# In[60]:


X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[61]:


#lets implemet the algorithm : Passive Aggressive Classifier
from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier(max_iter=50)

linear_clf.fit(X2_train, Y2_train)
prediction2 = linear_clf.predict(X2_test)
score = metrics.accuracy_score(Y2_test, prediction2)
print("accuracy:   %0.3f" % score)
cm2 = metrics.confusion_matrix(Y2_test, prediction2)
plot_confusion_matrix(cm2, classes=['FAKE Data', 'REAL Data'])


# # predictive system for passive agressive classifier

# In[64]:


X2_new = X2_test[1000]

prediction = model.predict(X2_new)
print(prediction)

if (prediction[0]==0):
  print('The news is fake')
else:
  print('The news is real')


# In[65]:


print(Y2_test[1000])


# # Classification report for Passive agressive classifier

# In[67]:


from sklearn.metrics import classification_report
print(classification_report(Y2_test,prediction2))


# # Random forest classifier

# In[103]:


X3_train, X3_test, Y3_train, Y3_test = train_test_split(X, Y, test_size=0.33, random_state=42, stratify=Y)


# In[104]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini',
                                 n_estimators=1000,
                                 random_state=1,
                                 n_jobs=2)


# In[105]:


# Fit the model
forest.fit(X3_train, Y3_train)


# In[106]:


# Measure model performance
Y_pred = forest.predict(X3_test)


# In[107]:


Y_pred


# In[108]:


print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(Y3_test, Y_pred)))


# In[109]:


# CONFUSION MATRIX
score = metrics.accuracy_score(Y3_test, Y_pred)
print("accuracy:   %0.3f" % score)
cm2 = metrics.confusion_matrix(Y3_test, Y_pred)
plot_confusion_matrix(cm2, classes=['FAKE Data', 'REAL Data'])


# #  predictive system for random forest classifier

# In[115]:


X3_new = X3_test[100]

prediction = model.predict(X3_new)
print(prediction)

if (prediction[0]==0):
  print('The news is FAKE')
else:
  print('The news is REAL')


# In[116]:


print(Y3_test[100])


# # Classification report for random forest classifier

# In[114]:


from sklearn.metrics import classification_report

print(classification_report(Y3_test, Y_pred))


# In[ ]:





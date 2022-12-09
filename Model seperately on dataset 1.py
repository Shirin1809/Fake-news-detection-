#!/usr/bin/env python
# coding: utf-8

# In[10]:


from platform import python_version


# In[11]:


print(python_version())


# In[12]:


import os
import numpy as np
import pandas as pd
import re


# In[13]:


pip install nltk


# In[14]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[15]:



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier


# In[16]:


from sklearn import metrics
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# In[17]:


import nltk
nltk.download('stopwords')


# In[18]:


news_dataset_train = pd.read_csv('Desktop/train.csv')


# In[19]:


news_dataset_train


# # Preprocessing the dataset
# 

# In[20]:


# shape of the dataset

news_dataset_train.shape


# In[21]:


# viewing first few rows of the dataset

news_dataset_train.head()


# In[22]:


# checking for missing values

news_dataset_train.isnull().sum()


# In[23]:


# replacing null values with empty string

news_dataset_train = news_dataset_train.fillna("")


# In[24]:


news_dataset_train


# In[25]:


# combine title and author
news_dataset_train['content'] = news_dataset_train['title'] + ' ' + news_dataset_train['author']


# In[26]:


print(news_dataset_train['content'])


# In[27]:


x= news_dataset_train[['title','label']]


# In[28]:


x.head()


# In[30]:


# Visualizing the count of 'Label' column from the dataset
import seaborn as sns
plt.figure(figsize=(8,8))
sns.countplot(x='label', data=x)
plt.xlabel('Classifier Real or Fake')
plt.ylabel('Count')
plt.show()


# # Stemming

# In[84]:


port_stem = PorterStemmer()


# In[85]:


# stemming function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[86]:


# applying stemming function
news_dataset_train['content'] = news_dataset_train['content'].apply(stemming)


# In[87]:


# viewing content after stemming
print(news_dataset_train['content'])


# In[94]:


#separating data and label

X = news_dataset_train['content'].values
Y = news_dataset_train['label'].values


# In[95]:


# converting textual data to numerical data

vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)


# In[96]:


print(X)


# #  Single word cloud

# In[91]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# initialize the word cloud
wordcloud = WordCloud( background_color='black', width=800, height=600)
# generate the word cloud by passing the news
text_cloud = wordcloud.generate(' '.join(news_dataset_train['text']))
# plotting the word cloud
plt.figure(figsize=(20,30))
plt.imshow(text_cloud)
plt.axis('off')
plt.show()


# In[42]:


from wordcloud import WordCloud

# make object of wordcloud
wc = WordCloud(background_color='black',min_font_size=10,width=500,height=500)

true_news_wc = wc.generate(news_dataset_train[news_dataset_train['label'] == 0]['content'].str.cat(sep=" "))
plt.figure(figsize=(8,6))
plt.imshow(true_news_wc)
plt.show()


# In[43]:


from wordcloud import WordCloud

# make object of wordcloud
wc = WordCloud(background_color='black',min_font_size=10,width=500,height=500)

fake_news_wc = wc.generate(news_dataset_train[news_dataset_train['label'] == 1]['content'].str.cat(sep=" "))
plt.figure(figsize=(8,6))
plt.imshow(fake_news_wc)
plt.show()


# # To find a specific count of the words in true and fake news

# In[97]:


# library for Count Words
from collections import Counter

# create list of True News words
true_news_words_list = (news_dataset_train[news_dataset_train['label'] == 0]['content'].str.cat(sep=" ")).split()

# create DataFrame of that
true_news_words_df = pd.DataFrame(Counter(true_news_words_list).most_common(20))

# Now Let's Plot barplot of this words
sns.barplot(x=true_news_words_df[0],y=true_news_words_df[1])
plt.xticks(rotation='vertical')
plt.xlabel('Words')
plt.ylabel('Counts')
plt.title('True News Words Count')
plt.show()


# In[98]:


# create list of Fake News words
fake_news_words_list = (news_dataset_train[news_dataset_train['label'] == 1]['content'].str.cat(sep=" ")).split()

# create DataFrame of that
fake_news_words_df = pd.DataFrame(Counter(fake_news_words_list).most_common(20))

# Now Let's Plot barplot of this words
sns.barplot(x=fake_news_words_df[0],y=fake_news_words_df[1])
plt.xticks(rotation='vertical')
plt.xlabel('Words')
plt.ylabel('Counts')
plt.title('Fake News Words Count')
plt.show()


# # Data spliting to test and train

# In[99]:


# splitting data to training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[103]:


X_train.shape


# In[104]:


X_test.shape


# In[106]:


Y_train.shape


# In[107]:


Y_test.shape


# # Model training using Logistic regression
# 

# In[108]:


model = LogisticRegression()


# In[109]:


model.fit(X_train, Y_train)


# In[115]:


# accuracy score on training data
prediction = model.predict(X_train)
accuracy_training_data = accuracy_score(prediction, Y_train)
print(f"Accuracy score for training data: {accuracy_training_data}")


# In[111]:


# accuracy score on test data
prediction = model.predict(X_test)
accuracy_test_data = accuracy_score(X_test_predictions, Y_test)
print(f"Accuracy score for test data: {accuracy_test_data}")


# # To build a predictive system for logistic regression

# In[116]:


# Build a Simple Predictive System

'''
index = int(input("Enter article number to be verified: "))
^ To get article number as input from user
'''

X_new = X_test[1]
new_predict = model.predict(X_new)
if(new_predict[0]==0):
    print("The News is real")
else:
    print("The News is fake")


# In[117]:


print(Y_test[1])


# # Classification report for logistic regression

# In[114]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, X_test_predictions))


# In[56]:


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


# In[59]:


cm1 = metrics.confusion_matrix(Y_test,X_test_predictions )
plot_confusion_matrix(cm1, classes=['FAKE', 'REAL'])


# # Classification Model: Multinomial Naive Bayes

# In[61]:


#let's implement the model : Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


# In[62]:


X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[63]:


X1_train


# In[64]:


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


# In[65]:


from sklearn import metrics
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

classifier.fit(X1_train, Y1_train)
prediction1 = classifier.predict(X1_test)
score = metrics.accuracy_score(Y1_test, prediction1)
print("accuracy:   %0.3f" % score)
cm1 = metrics.confusion_matrix(Y1_test, prediction1)
plot_confusion_matrix(cm1, classes=['FAKE data', 'REAL data'])


# # Predictive system for Multinomial Naive Bayes

# In[66]:


X1_new = X1_test[600]

prediction = model.predict(X1_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')


# In[67]:


print(Y_test[100])


# # Classification report for Multinomial Naive Bayes

# In[68]:


from sklearn.metrics import classification_report
print(classification_report(Y1_test,prediction1))


# # Passive Agressive classifier

# In[69]:


X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[70]:


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

# In[71]:


X2_new = X2_test[1000]

prediction = model.predict(X2_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')


# In[72]:


print(Y2_test[100])


# # Classification report for Passive agressive classifier

# In[73]:


from sklearn.metrics import classification_report
print(classification_report(Y2_test,prediction2))


# # Random forest classifier
# 

# In[74]:


X3_train, X3_test, Y3_train, Y3_test = train_test_split(X, Y, test_size=0.33, random_state=42, stratify=Y)


# In[75]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini',
                                 n_estimators=100,
                                 random_state=1,
                                 n_jobs=2)


# In[76]:


# Fit the model
forest.fit(X3_train, Y3_train)


# In[77]:


# Measure model performance
Y_pred = forest.predict(X3_test)
print('Accuracy: %.3f' % accuracy_score(Y3_test, Y_pred))


# In[78]:


#CONFUSION MATRIX
score = metrics.accuracy_score(Y3_test, Y_pred)
print("accuracy:   %0.3f" % score)
cm2 = metrics.confusion_matrix(Y3_test, Y_pred)
plot_confusion_matrix(cm2, classes=['FAKE Data', 'REAL Data'])


# # predictive system for random forest classifier¶

# In[79]:


X3_new = X3_test[100]

prediction = model.predict(X3_new)
print(prediction)

if (prediction[0]==0):
  print('The news is REAL')
else:
  print('The news is Fake')


# In[80]:


print(Y[100])


# # Classification report for Random Forest Classifier

# In[118]:


from sklearn.metrics import classification_report

print(classification_report(Y3_test, Y_pred))


# In[ ]:





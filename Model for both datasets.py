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


#reading the dataset 1
ds1 = pd.read_csv('Desktop/train.csv')


# In[3]:


ds1


# In[4]:


#reading the dataset 2
ds2= pd.read_csv('Desktop/WELFake_Dataset.csv')


# In[5]:


ds2


# In[6]:


ds1.head()


# In[7]:


ds2.head()


# In[8]:


ds2= ds2[0:20800]


# In[9]:


ds2


# # Pre-processing of the data

# In[10]:


# shape of the dataset

ds1.shape


# In[11]:


ds2.shape


# In[12]:


# checking for missing values

ds1.isnull().sum()


# In[13]:


#cheking for the missing values in other dataset

ds2.isnull().sum()


# In[14]:


# replacing null values with empty string

ds1= ds1.fillna("")


# In[15]:


ds1


# In[16]:


ds2= ds2.fillna("")
ds2


# In[17]:


ds1= ds1.fillna('')
ds1


# In[18]:


# combine text and title so that it will be easy to understand what title has caused the reason fake and by which author it caused
ds1['content'] = ds1['title'] + ' ' + ds1['text']
ds1['content']


# In[19]:


#same for the second dataset
ds2['content'] = ds2['title'] + ' ' + ds2['text']
ds2['content']


# In[20]:


#To retrieve only the title and its label
x= ds1[['title','label']]


# In[21]:


x


# In[22]:


y=ds2[['title','label']]
y


# # Exploratory data analysis

# In[23]:


#For the dataset 1
# Visualizing the count of 'Label' column from the dataset
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
sns.countplot(x='label', data=x)
plt.xlabel('Classifier 1= fake, 0= real')
plt.ylabel('Count')
plt.show()


# In[24]:


#For the dataset 2
# Visualizing the count of 'Label' column from the dataset
import seaborn as sns
plt.figure(figsize=(8,8))
sns.countplot(x='label', data=y)
plt.xlabel('Classifier 1=Real or 0=Fake')
plt.ylabel('Count')
plt.show()


# In[25]:


#creating a word cloud for dataset 12
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# initialize the word cloud
wordcloud = WordCloud( background_color='black', width=800, height=600)
# generate the word cloud by passing the news
text_cloud = wordcloud.generate(' '.join(ds1['text']))
# plotting the word cloud
plt.figure(figsize=(20,30))
plt.imshow(text_cloud)
plt.axis('off')
plt.show()


# In[26]:


#Creating a word cloud for dataset 2
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# initialize the word cloud
wordcloud = WordCloud( background_color='black', width=800, height=600)
# generate the word cloud by passing the news
text_cloud = wordcloud.generate(' '.join(ds2['text']))
# plotting the word cloud
plt.figure(figsize=(20,30))
plt.imshow(text_cloud)
plt.axis('off')
plt.show()


# In[27]:


#to find the specific counts of words in fake and real in dataset1
# library for Count Words
from collections import Counter

# create list of True News words
true_news = (ds1[ds1['label'] == 0]['content'].str.cat(sep=" ")).split()

# create DataFrame of that
true_news_df = pd.DataFrame(Counter(true_news).most_common(20))

# Now Let's Plot barplot of this words
sns.barplot(x=true_news_df[0],y=true_news_df[1])
plt.xticks(rotation='vertical')
plt.xlabel('Words')
plt.ylabel('Counts')
plt.title('True News Words Count')
plt.show()


# In[29]:


#to find the specific counts of words in fake and real in dataset1
# library for Count Words
from collections import Counter

# create list of True News words
fake_news = (ds1[ds1['label'] == 1]['content'].str.cat(sep=" ")).split()

# create DataFrame of that
fake_news_df = pd.DataFrame(Counter(fake_news).most_common(20))

# Now Let's Plot barplot of this words
sns.barplot(x=fake_news_df[0],y=fake_news_df[1])
plt.xticks(rotation='vertical')
plt.xlabel('Words')
plt.ylabel('Counts')
plt.title('Fake News Words Count')
plt.show()


# In[30]:


#to find the specific counts of words in fake and real in dataset2
# library for Count Words
from collections import Counter

# create list of True News words
true_news1 = (ds2[ds2['label'] == 1]['content'].str.cat(sep=" ")).split()

# create DataFrame of that
true_news_df1 = pd.DataFrame(Counter(true_news1).most_common(20))

# Now Let's Plot barplot of this words
sns.barplot(x=true_news_df1[0],y=true_news_df1[1])
plt.xticks(rotation='vertical')
plt.xlabel('Words')
plt.ylabel('Counts')
plt.title('True News Words Count')
plt.show()


# In[31]:


#to find the specific counts of words in fake and real in dataset2
# library for Count Words
from collections import Counter

# create list of True News words
fake_news1 = (ds2[ds2['label'] == 0]['content'].str.cat(sep=" ")).split()

# create DataFrame of that
fake_news_df1 = pd.DataFrame(Counter(fake_news1).most_common(20))

# Now Let's Plot barplot of this words
sns.barplot(x=fake_news_df1[0],y=fake_news_df1[1])
plt.xticks(rotation='vertical')
plt.xlabel('Words')
plt.ylabel('Counts')
plt.title('Fake News Words Count')
plt.show()


# # Stemming and removal of stop words

# In[32]:


port_stemmer = PorterStemmer()


# In[33]:


# stemming function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[34]:


# applying stemming function for dataset 1
ds1['content'] = ds1['content'].apply(stemming)


# In[37]:


# applying stemming function for dataset 2
ds2['content'] = ds2['content'].apply(stemming)


# In[38]:


# viewing content after stemming for dataset1
print(ds1['content'])


# In[39]:


# viewing content after stemming for dataset1
print(ds2['content'])


# In[40]:


#separating data and label to get the values for dataset1

X = ds1['content'].values
Y = ds1['label'].values


# In[41]:


print(X)


# In[42]:


Y


# In[43]:


#separating data and label to get the values for dataset 2

X1 = ds2['content'].values
Y1 = ds2['label'].values


# In[44]:


X1


# In[45]:


Y1


# In[46]:


# converting textual data to numerical data by using TFIDF for dataset1

vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X_1 = vectorizer.transform(X)


# In[47]:


print(X_1)


# In[48]:


# converting textual data to numerical data for dataset 2

vectorizer1 = TfidfVectorizer()
vectorizer1.fit(X1)

X_2= vectorizer1.transform(X1)


# In[49]:


print(X_2)


# # Data splitting for testing and training

# In[50]:


# splitting data to training and testing for dataset1
X_train, X_test, Y_train, Y_test = train_test_split(X_1, Y, test_size=0.2, random_state=7)


# In[51]:


# splitting data to training and testing for dataset 2
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X_2, Y1, test_size=0.2, random_state=7)


# In[52]:


X_train.shape


# In[53]:


X_test.shape


# In[54]:


Y_train.shape


# In[55]:


Y_test.shape


# In[56]:


print(X1_train.shape)


# In[57]:


print(X1_test.shape)


# In[58]:


Y1_train.shape


# In[59]:


Y1_test.shape


# # Logistic regression model on dataset 1

# In[60]:


model = LogisticRegression()


# In[61]:


model.fit(X_train, Y_train)


# In[62]:


# accuracy score on test data
prediction = model.predict(X_test)
accuracy_test_data = accuracy_score(prediction, Y_test)
print(f"Accuracy score for test data: {accuracy_test_data}")


# In[63]:


#assigning a function to plot confusion matrix
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


# In[64]:


from sklearn import metrics
import itertools
from sklearn.metrics import confusion_matrix
cm1= metrics.confusion_matrix(Y_test,prediction )
plot_confusion_matrix(cm1, classes=['fake data', 'real data'])


# In[65]:


print(cm1)


# In[66]:


#To get a classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test,prediction))


# # Logistic regression for dataset 2

# In[67]:


model_1 = LogisticRegression()


# In[68]:


model_1.fit(X1_train, Y1_train)


# In[69]:


# accuracy score on test data
prediction_1 = model_1.predict(X1_test)
accuracy_test_data_1 = accuracy_score(prediction_1, Y1_test)
print(f"Accuracy score for test data: {accuracy_test_data_1}")


# In[70]:


from sklearn import metrics
import itertools
from sklearn.metrics import confusion_matrix
cm_1= metrics.confusion_matrix(Y1_test,prediction_1 )
plot_confusion_matrix(cm1, classes=['FAKE', 'REAL'])


# In[71]:


print(cm_1)


# In[73]:


#To get a classification report
from sklearn.metrics import classification_report
print(classification_report(Y1_test,prediction_1))


# # Passive Agressive algorithm for dataset 1

# In[75]:


#lets implemet the algorithm : Passive Aggressive Classifier
from sklearn.linear_model import PassiveAggressiveClassifier
classifier = PassiveAggressiveClassifier(max_iter=50)


# In[76]:


classifier.fit(X_train, Y_train)
prediction_2 = classifier.predict(X_test)
score = metrics.accuracy_score(Y_test, prediction_2)
print("accuracy:   %0.3f" % score)
cm_2 = metrics.confusion_matrix(Y_test, prediction_2)
plot_confusion_matrix(cm_2, classes=['FAKE Data', 'REAL Data'])


# In[77]:


#To get a classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test,prediction_2))


# # Passive agressive classifier for dataset 2

# In[78]:


#lets implemet the algorithm : Passive Aggressive Classifier
from sklearn.linear_model import PassiveAggressiveClassifier
classifier_1 = PassiveAggressiveClassifier(max_iter=50)


# In[80]:


classifier_1.fit(X1_train, Y1_train)
prediction_3 = classifier_1.predict(X1_test)
score = metrics.accuracy_score(Y1_test, prediction_3)
print("accuracy:   %0.3f" % score)
cm_3 = metrics.confusion_matrix(Y1_test, prediction_3)
plot_confusion_matrix(cm_3, classes=['FAKE Data', 'REAL Data'])


# # Multinomial naive bayes for dataset 1

# In[82]:


#let's implement the model : Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
classifier_2=MultinomialNB()


# In[86]:


classifier_2.fit(X_train, Y_train)
prediction_4 = classifier_2.predict(X_test)
score = metrics.accuracy_score(Y_test, prediction_4)
print("accuracy:   %0.3f" % score)
cm_4 = metrics.confusion_matrix(Y_test, prediction_4)
plot_confusion_matrix(cm_4, classes=['FAKE data', 'REAL data'])


# In[87]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,prediction_4))


# # Multinomial naive bayes for dataset2

# In[88]:


#let's implement the model : Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
classifier_3=MultinomialNB()


# In[89]:


classifier_3.fit(X1_train, Y1_train)
prediction_5 = classifier_3.predict(X1_test)
score = metrics.accuracy_score(Y1_test, prediction_5)
print("accuracy:   %0.3f" % score)
cm_5 = metrics.confusion_matrix(Y1_test, prediction_5)
plot_confusion_matrix(cm_5, classes=['FAKE data', 'REAL data'])


# In[91]:


from sklearn.metrics import classification_report
print(classification_report(Y1_test,prediction_5))


# # Random forest classifier for dataset 1

# In[102]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini',
                                 n_estimators=100,
                                 random_state=1,
                                 n_jobs=2)


# In[103]:


# Fit the model
forest.fit(X_train, Y_train)


# In[104]:


# Measure model performance
prediction_6 = forest.predict(X_test)
print('Accuracy: %.3f' % accuracy_score(Y_test, prediction_6))


# In[105]:


cm_6 = metrics.confusion_matrix(Y_test, prediction_6)
plot_confusion_matrix(cm_6, classes=['FAKE data', 'REAL data'])


# In[106]:


from sklearn.metrics import classification_report

print(classification_report(Y_test, prediction_6))


# # Random forest classifier on dataset 2

# In[107]:


from sklearn.ensemble import RandomForestClassifier
forest_1 = RandomForestClassifier(criterion='gini',
                                 n_estimators=100,
                                 random_state=1,
                                 n_jobs=2)


# In[108]:


forest_1.fit(X1_train, Y1_train)


# In[109]:


# Measure model performance
prediction_7 = forest_1.predict(X1_test)
print('Accuracy: %.3f' % accuracy_score(Y1_test, prediction_7))


# In[110]:


cm_7 = metrics.confusion_matrix(Y1_test, prediction_7)
plot_confusion_matrix(cm_7, classes=['FAKE data', 'REAL data'])


# In[112]:


from sklearn.metrics import classification_report

print(classification_report(Y1_test, prediction_7))


# In[ ]:





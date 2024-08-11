import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

dataset = pd.read_csv("/Users/karthikeyakammili/Downloads/Data Sets/ML/BBC_News_processed.csv")

# print(dataset['Category'].value_counts())
target_category = dataset['Category'].unique()
# print(target_category)
dataset['CategoryId'] = dataset['Category'].factorize()[0]
# print(dataset.head(10))

X = dataset['Text']
y = dataset['Category_target']
# print(X)
# print(y)
# print(f'Null values :{X.isnull().sum().sum()}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train_Count = vectorizer.fit_transform(X_train.values)
# print(X_train_Count.toarray())

X_test_count = vectorizer.transform(X_test)


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_Count, y_train)

from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train_Count, y_train)


emails = [" WHY ROHIDAS GOT A RED CARD The 31-year-old Rohidas was spotted swinging his stick across the face of British forward Will Calnan in the second quarter of the match on Sunday. The on-field umpire did not rule the offence as serious to get a red card, but Rohidas was shown red and given the marching orders after a video referral. Subsequently, India played the British team with 10 players for more than 40 minutes. However, the Harmanpreet Singh-led side defended heroically, thanks largely to PR Sreejesh's presence in the goalpost, to hold onto the 1-1 scoreline and take the game to a shootout. Goalkeeper Sreejesh once again turned out to be the star, saving two attempts by the GB team in the one-on-one shootout to help India earn a dramatic 4-2 win and a second successive entry into the semis at the Olympics."]
emails_count = vectorizer.transform(emails)

print(model.predict(emails_count))
print(svm_model.predict(emails_count))

print(f'Accuracy by Navie Bayes model: {model.score(X_test_count, y_test)}')
print(f'Accuracy by SVM model: {svm_model.score(X_test_count, y_test)}')


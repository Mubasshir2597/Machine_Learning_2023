import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import missingno as msno

import pymongo
from pymongo import *
import pandas as pd
import json


#Checking whether the connection 
try:
    Connection = MongoClient("mongodb+srv://Muba:2597@lab7.u5asnmz.mongodb.net/?retryWrites=true&w=majority")
    #client = pymongo.MongoClient(connect=True, serverSelectionTimeoutMS=5000)
    print("Connected to the Server")
    #print(client.server_info())
except Exception:
    print( "Connection to the Server is unsuccessful")

# Reading the CSV file and converting it into Json File C:\Users\techn\OneDrive\Desktop\diab
myTestingCSVData  = pd.read_csv(r"C:/Users/techn/OneDrive/Desktop/diab/diabete.csv")
#print(myTestingCSVData.head(10))
myTestingCSVData.to_json("C:/Users/techn/OneDrive/Desktop/diab/diabete.json")
#myTestingJSonData = open(**r"C:\Users\techn\OneDrive\Desktop\diab\diabete.json"**)
myTestingJSonData = open(r"C:/Users/techn/OneDrive/Desktop/diab/diabete.json")
jsonData = json.load(myTestingJSonData)

# inserting the Json data to the mongoDB
DB  = Connection["trial_db"]
Col = DB["Testing_Collection"]
Col.insert_many([jsonData])
print("Json Data Inserted")

# Dropping the Created Collection
#Col.drop()
print(f'The collection {Col} is dropped sucessfully')

col_list = DB.list_collection_names()
for i in col_list:
    print(i)

df = pd.read_csv(r"C:/Users/techn/OneDrive/Desktop/diab/diabete.csv")
df.to_json (r"C:/Users/techn/OneDrive/Desktop/diab/diabete.json")
df.head()

# see the column names and its datatypes
df.info()

#shape
df.shape

df.describe()

#There is a huge variation in mean, and we can see there's no missing values, but for some of the columns like Glucose , BP, Skin Thickness,BMI has 0 as min value, which is not possible, hence we can treat this as missingvalues and impute accordingly.
features = df.columns
cols = (df[features] == 0).sum()
print(cols)


#We cannot drop these values, as our data is very small. So let's handle them.
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df.isnull().sum()

#Handle: Glucose, BloodPressure, BMI
#Replace the null values with the median of that column:
df['Glucose'].fillna(df['Glucose'].median(), inplace =True)
df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace =True)
df['BMI'].fillna(df['BMI'].median(), inplace =True)


#Handle: Insulin based on Glucose
by_Glucose_Age_Insulin_Grp = df.groupby(['Glucose'])
def fill_Insulin(series):
    return series.fillna(series.median())
df['Insulin'] = by_Glucose_Age_Insulin_Grp['Insulin'].transform(fill_Insulin)
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())


#Skinthickness with respect to BMI
by_BMI_Insulin = df.groupby(['BMI'])

def fill_Skinthickness(series):
    return series.fillna(series.mean())
df['SkinThickness'] = by_BMI_Insulin['SkinThickness'].transform(fill_Skinthickness)
df['SkinThickness'].fillna(df['SkinThickness'].mean(),inplace= True)
df.isnull().sum()



                       #Visualization of Target Variable
import matplotlib.style as style
style.available

style.use('seaborn-pastel')
labels = ["Healthy", "Diabetic"]
df['Outcome'].value_counts().plot(kind='pie',labels=labels, subplots=True,autopct='%1.0f%%', labeldistance=1.2, figsize=(9,9))

from matplotlib.pyplot import figure, show

figure(figsize=(8,6))
ax = sns.countplot(x=df['Outcome'], data=df,palette="husl")
ax.set_xticklabels(["Healthy","Diabetic"])
healthy, diabetics = df['Outcome'].value_counts().values
print("Samples of diabetic people: ", diabetics)
print("Samples of healthy people: ", healthy)
#Distribution of other features w.r.t Outcome


#Distribution of Pregnancies
plt.figure()
ax = sns.distplot(df['Pregnancies'][df.Outcome == 1], color ="darkturquoise", rug = True)
sns.distplot(df['Pregnancies'][df.Outcome == 0], color ="lightcoral",rug = True)
plt.legend(['Diabetes', 'No Diabetes'])


#Distribution of Glucose
plt.figure()
ax = sns.distplot(df['Glucose'][df.Outcome == 1], color ="darkturquoise", rug = True)
sns.distplot(df['Glucose'][df.Outcome == 0], color ="lightcoral", rug = True)
plt.legend(['Diabetes', 'No Diabetes'])

#Distribution of BloodPressure
plt.figure()
ax = sns.distplot(df['BloodPressure'][df.Outcome == 1], color ="darkturquoise", rug=True)
sns.distplot(df['BloodPressure'][df.Outcome == 0], color ="lightcoral", rug=True)
plt.legend(['Diabetes', 'No Diabetes'])


#Distribution of SkinThickness
plt.figure()
ax = sns.distplot(df['SkinThickness'][df.Outcome == 1], color ="darkturquoise", rug=True)
sns.distplot(df['SkinThickness'][df.Outcome == 0], color ="lightcoral", rug=True)
plt.legend(['Diabetes', 'No Diabetes'])

#Distribution of Insulin
plt.figure()
ax = sns.distplot(df['Insulin'][df.Outcome == 1], color ="darkturquoise", rug=True)
sns.distplot(df['Insulin'][df.Outcome == 0], color ="lightcoral", rug=True)
plt.legend(['Diabetes', 'No Diabetes'])


#Distribution of BMI
plt.figure()
ax = sns.distplot(df['BMI'][df.Outcome == 1], color ="darkturquoise", rug=True)
sns.distplot(df['BMI'][df.Outcome == 0], color ="lightcoral", rug=True)
plt.legend(['Diabetes', 'No Diabetes'])



#Distribution of DiabetesPedigreeFunction
plt.figure()
ax = sns.distplot(df['DiabetesPedigreeFunction'][df.Outcome == 1], color ="darkturquoise", rug=True)
sns.distplot(df['DiabetesPedigreeFunction'][df.Outcome == 0], color ="lightcoral", rug=True)
plt.legend(['Diabetes', 'No Diabetes'])


#Distribution of Age
plt.figure()
ax = sns.distplot(df['Age'][df.Outcome == 1], color ="darkturquoise", rug=True)
sns.distplot(df['Age'][df.Outcome == 0], color ="lightcoral", rug=True)
sns.distplot(df['Age'], color ="green", rug=True)
plt.legend(['Diabetes', 'No Diabetes', 'all'])


#Correlation Matrix
plt.figure(dpi = 120,figsize= (5,4))
mask = np.triu(np.ones_like(df.corr(),dtype = bool))
sns.heatmap(df.corr(),mask = mask, fmt = ".2f",annot=True,lw=1,cmap = 'plasma')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.title('Correlation Heatmap')
plt.show()

#Pair Plot
sns.pairplot(df, hue="Outcome",palette="husl")


#MODEL BUILDING -->Split the data into test & train

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

print("Number transactions x_train dataset: ", x_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions x_test dataset: ", x_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

#Feature Scaling
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,accuracy_score,auc

#SVM MODEL
#Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.
#The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.
from sklearn.svm import SVC

model=SVC(kernel='rbf')
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
accuracy_score(y_test,y_pred)


print(classification_report(y_test,y_pred))


fpr,tpr,_=roc_curve(y_test,y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

#Random Forest Model
#Random forest classifier creates a set of decision trees from randomly selected subset of training set. It then aggregates the votes from different decision trees to decide the final class of the test object.
#This works well because a single decision tree may be prone to a noise, but aggregate of many decision trees reduce the effect of noise giving more accurate results.

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier()
classifier.fit(x_train,y_train)


Y_pred=classifier.predict(x_test)
accuracy_score(y_test,Y_pred)

print(classification_report(y_test,Y_pred))


fpr,tpr,_=roc_curve(y_test,Y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
#print('ROC AUC: %0.2f' % roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


#KNN Model
#K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique. It can be used for Regression as well as for Classification but mostly it is used for the Classification problems.
#K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3) 
clf.fit(x_train,y_train)  
print(clf.score(x_test,y_test))

y_pred=clf.predict(x_test)
print(classification_report(y_test,y_pred))


fpr,tpr,_=roc_curve(y_test,y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
#print('ROC AUC: %0.2f' % roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# Conclusion
#In this kernel, I have performed Exploratory Data Analysis, Data Preprocessing, Visualization of Features, Correlation Matrix, Model Building (SVM, RF, KNN).
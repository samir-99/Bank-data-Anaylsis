#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules 


# In[3]:


data = pd.read_csv('project_data.csv')
data.head()

datac=[]
for x in range(len(data)):
    ex=str(list(data.iloc[x].values)[0]).split(';')
    datac.append(ex)

datac=np.array(datac)
datac


# In[ ]:





# In[ ]:





# In[4]:


columns =["Status of existing checking account of the customer",
        "Duration of the credit in month",
        "Credit history of the customer",
        "Purpose of the credit",
        "Credit amount in EUR",
        "Savings account/bonds of the customer",
        "Present employment of teh customer since",
        "Installment rate in percentage of disposable income",
        "Personal status and sex of the customer",
        "Other debtors or guarantors for the credit",
        "Present residence of the customer since (in years)",
               "Property owned by the customer",
               "Age of the customer in years",
               "Other installment plans of the customer",
               "Housing situation of the customer",
               "Number of existing credits of the customer at this bank",
                  "Job situation of the customer",
          
               "Number of people the customer being liable to provide maintenance for",
               "Telephone of the customer",
               "customer is a foreign worker",
               "label"]


# In[ ]:





# In[5]:


df=pd.DataFrame(data=datac, columns=columns)
df


# In[ ]:





# In[6]:


c1= {'A11': 'X01 < 0 EUR',
'A12': '0 <= X01 < 200 EUR',
'A13': 'X01 >= 200 EUR',
'A14': 'no checking account' }
df.replace(c1,inplace=True)

c3={'A30': 'no credits taken or all credits paid back duly',
'A31': 'all credits at this bank paid back duly',
'A32': 'existing credits paid back duly till now',
'A33': 'delay in paying off in the past',
'A34': 'critical account or other credits existing (not at this bank)'}
df.replace(c3,inplace=True)

c4={'A40': 'car (new)',
    'A41': 'car (used)',
    'A42': 'furniture/equipment',
    'A43': 'radio/television',
    'A44': 'domestic appliances',
    'A45': 'repairs',
    'A46': 'education',
    'A47': 'vacation',
    'A48': 'retraining',
    'A49': 'business',
    'A410': 'others',
}
df.replace(c3,inplace=True)

c6={'A61': 'X06 < 100 EUR',
'A62': '100 <= X06 < 500 EUR',
'A63': '500 <= X06 < 1000 EUR',
'A64': 'X06 >= 1000 EUR',
'A65': 'unknown/no savings account'}

df.replace(c6,inplace=True)
    
c7={'A71': 'unemployed',
    'A72': 'X07 < 1 year',
    'A73': '1 <= X07 < 4 years',  
    'A74': '4 <= X07 < 7 years',
    'A75': 'X07 >= 7 years'}
df.replace(c7,inplace=True)
    
c9={'A91': 'male - divorced/separated',
'A92': 'female - divorced/separated/married',
'A93': 'male - single',
'A94': 'male - married/widowed',
'A95': 'female - single'}
    
df.replace(c9,inplace=True)

c10={'A101': 'none',
'A102': 'co-applicant',
'A103': 'guarantor'}
    
df.replace(c10,inplace=True)
    
c12={'A121': 'real estate',
'A122': 'building society savings agreement/life insurance',
'A123': 'car or other',
'A124': 'unknown/no property'}

df.replace(c12,inplace=True)
    
c14={'A141': 'bank',
'A142': 'stores',
'A143': 'none'}
    
df.replace(c14,inplace=True)
    
c15={'A151': 'renting',
'A152': 'owning',
'A153': 'accommodation (ie. living) for free'}
df.replace(c15,inplace=True)
    
c17={'A171': 'unemployed/unskilled  - non-resident',
'A172': 'unskilled - resident',
'A173': 'skilled employee/official',
'A174': 'management/self-employed/highly qualified employee/officer'}

c19={'A191': "no",
     'A192': 'yes'}
    
c20={"A201": "yes",
'A202': "no"}

df.replace(c17,inplace=True) 
df.replace(c19,inplace=True)
df.replace(c20,inplace=True)
dft=df

df.isnull().sum()


# In[ ]:





# In[7]:


#Convert objects to numeric
df['Duration of the credit in month']=df['Duration of the credit in month'].astype(str).astype(int)
df['Credit amount in EUR']=df['Credit amount in EUR'].astype(str).astype(float)
df['Installment rate in percentage of disposable income']=df['Installment rate in percentage of disposable income'].astype(str).astype(int)
df['Age of the customer in years']=df['Age of the customer in years'].astype(str).astype(int)
df['Number of existing credits of the customer at this bank']=df['Number of existing credits of the customer at this bank'].astype(str).astype(int)
df['Number of people the customer being liable to provide maintenance for']=df['Number of people the customer being liable to provide maintenance for'].astype(str).astype(int)
df['Present residence of the customer since (in years)']=df['Present residence of the customer since (in years)'].astype(str).astype(int)


# In[ ]:





# In[8]:


#EDA
df.hist(bins=30,figsize=(30,30))
plt.show()


# In[9]:


corr=df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[10]:


df['Number of people the customer being liable to provide maintenance for'].value_counts().plot(kind='bar');


# In[11]:


df['Other debtors or guarantors for the credit'].value_counts().plot(kind='bar');


# In[12]:


df['customer is a foreign worker'].value_counts().plot(kind='bar');


# In[13]:


df["Other installment plans of the customer"].value_counts().plot(kind='bar')


# In[14]:


#Preprocessing
fpm=df.drop(columns=["Telephone of the customer",
                     "Other debtors or guarantors for the credit",
                     "Other installment plans of the customer",
                     "customer is a foreign worker",
                     'Duration of the credit in month',
                      'Credit amount in EUR',
                      'Age of the customer in years',
                      'Number of existing credits of the customer at this bank',
                      'Number of people the customer being liable to provide maintenance for',
                      'Present residence of the customer since (in years)',
                      'Installment rate in percentage of disposable income',
                       'label' ])


# In[15]:


fpm=pd.get_dummies(fpm,fpm.columns)


# In[16]:


# Convert categorical values to numeric
df=df.drop("Telephone of the customer",axis=1)
cat_col=['Status of existing checking account of the customer',
         'Credit history of the customer',
         'Purpose of the credit',
         'Savings account/bonds of the customer',
         'Present employment of teh customer since',
         'Personal status and sex of the customer',
         'Other debtors or guarantors for the credit',
         'Property owned by the customer',
         'Other installment plans of the customer',
         'Housing situation of the customer',
         'Job situation of the customer',
         'customer is a foreign worker']
df = pd.get_dummies(df, columns=cat_col)


# In[17]:


df['label'].replace({'2':0,'1':1},inplace=True)
df


# In[18]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df)
datanor = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
datanor.describe()


# In[ ]:





# In[19]:


#Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
dfc=df.drop('label',axis=1)
scores=[]
for x in range(2,20):
        kmeans=KMeans(n_clusters=x)
        kmeans.fit(datanor)
        score = silhouette_score(datanor, kmeans.labels_, metric='euclidean')
        scores.append(score)

scores


# In[20]:


df.describe()


# In[21]:




dfc=df.drop('label',axis=1)
scores=[]
for x in range(2,20):
        kmeans=KMeans(n_clusters=x)
        kmeans.fit(dfc)
        score = silhouette_score(dfc, kmeans.labels_, metric='euclidean')
        scores.append(score)

scores
plt.plot(range(2,20),scores)
plt.title('The Silhoutte scores')
plt.xlabel('Number of clusters')
plt.ylabel('Scores')
plt.show()


# In[ ]:





# In[ ]:





# In[22]:


# When we normalize data it resulted in bad silhoutte score.


# In[23]:


wcss=[]
for i in range(1,10): 
     kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )
     kmeans.fit(dfc)
     wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,8))
plt.plot(range(1,10),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[24]:


from sklearn.cluster import AgglomerativeClustering
scores=[]
for x in range(2,20):
        kmeans=AgglomerativeClustering(n_clusters=x)
        kmeans.fit(dfc)
        score = silhouette_score(dfc, kmeans.labels_, metric='euclidean')
        scores.append(score)

scores

plt.plot(range(2,20),scores)
plt.title('The Silhoutte scores')
plt.xlabel('Number of clusters')
plt.ylabel('Scores')
plt.show()


# In[ ]:





# In[25]:


# So number of clusters are 2 in this dataset
kmeans=KMeans(n_clusters=2)
kmeans.fit(dfc)
dfc['labels of cluster']=kmeans.labels_
dfc


# In[ ]:





# In[26]:


dfc.groupby("labels of cluster").mean()


# In[27]:


from sklearn.model_selection import train_test_split
np.random.seed(100)

X=df.drop('label',axis=1)
y=df['label']
X_train, X_test,y_train, y_test= train_test_split(X,y,test_size=0.2)


# In[28]:


# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score




model1= LogisticRegression()
model1.fit(X_train,y_train)
y_pred1= model1.predict(X_test)
cm1=confusion_matrix(y_test, y_pred1)
cm1


# In[29]:


print("Accuracy: " ,round(cross_val_score(model1,X,y,cv=5).mean()*100,2) , "%")


# In[30]:


cross_val_score(model1,X,y,cv=5)


# In[31]:


print(classification_report(y_test,y_pred1))


# In[32]:


from sklearn.neighbors import KNeighborsClassifier

model2= KNeighborsClassifier()
model2.fit(X_train,y_train)
y_pred2= model2.predict(X_test)
cm1=confusion_matrix(y_test, y_pred2)
cm1


# In[33]:


accuracy_score(y_test,y_pred2)


# In[34]:


accs=[]
for x in range (2,50):
    model2= KNeighborsClassifier(n_neighbors=x)
    model2.fit(X_train,y_train)
    y_pred2= model2.predict(X_test)
    accs.append(accuracy_score(y_test,y_pred2))

plt.plot(range(2,50),accs)
plt.title('Accuracy scores')
plt.xlabel('Number of neighbours')
plt.ylabel('Scores')
plt.show()
print(np.argmax(np.array(accs))+2)
print(accs[np.argmax(np.array(accs))])


# In[35]:


model2= KNeighborsClassifier(n_neighbors=np.argmax(np.array(accs))+2)
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)
accuracy_score(y_test,y_pred2)


# In[36]:


print("Accuracy: " ,round(cross_val_score(model2,X,y,cv=5).mean()*100,2) , "%")


# In[37]:


cm2=confusion_matrix(y_test, y_pred2)
cm2


# In[38]:


print(classification_report(y_test,y_pred2))


# In[ ]:





# In[39]:


cross_val_score(model2,X,y,cv=5)


# In[ ]:





# In[40]:


scaler = MinMaxScaler()
scaler.fit(df)
datanor = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)


# In[41]:


Xn=df.drop('label',axis=1)
yn=df['label']

X_trainn, X_testn,y_trainn, y_testn= train_test_split(Xn,yn,test_size=0.2)
np.random.seed(99)
print("Accuracy(LR): " ,round(cross_val_score(model1,X,y,cv=5).mean()*100,2) , "%")
print("Accuracy(KNN): " ,round(cross_val_score(model2,X,y,cv=5).mean()*100,2) , "%")


# In[ ]:





# In[42]:


print("Cross validation score(LR): " ,round(cross_val_score(model1,Xn,yn,cv=5).mean()*100,2) , "%")
print("Cross validationn score(KNN): " ,round(cross_val_score(model2,Xn,yn,cv=5).mean()*100,2) , "%")


# In[ ]:





# In[43]:


from sklearn.tree import DecisionTreeClassifier
accuracy_scores = []
max_depths = []
cvs=[]

for max_depth in range(1, 16):
    model3 = DecisionTreeClassifier(max_depth = max_depth)
    model3.fit(X_train, y_train)

    test_prediction = model3.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_prediction)
    
    max_depths.append(max_depth)
    accuracy_scores.append(test_accuracy)
    cvs.append(round(cross_val_score(model3,X,y,cv=5).mean()*100,2))

ax = sns.lineplot(x = max_depths, y = cvs)
ax.set(xlabel='Decision Tree Max Depth', ylabel='Accuracy Score')
print(f'Best accuracy score |{max(cvs)}| achieved at max depth |{np.argmax(cvs) + 1}|')

model3 = DecisionTreeClassifier(max_depth=max_depths[np.argmax(cvs)])
model3.fit(X_train, y_train)


# In[44]:


y_pred3=model3.predict(X_test)

print(classification_report(y_test,y_pred3))


# In[45]:


confusion_matrix(y_test,y_pred3)


# In[46]:


from sklearn.ensemble import RandomForestClassifier
modelrfc=RandomForestClassifier(n_estimators=100)
modelrfc.fit(X_train,y_train)
yrfc=modelrfc.predict(X_test)
accuracy_score(y_test,yrfc)


# In[47]:


rfccvs=(round(cross_val_score(modelrfc,X,y,cv=5).mean()*100,2))
rfccvs


# In[48]:


accuracy_scores = []
max_depths = []
cvs=[]
np.random.seed(99)
for estimator in range(1, 150):
    modelrfc=RandomForestClassifier(n_estimators=estimator)
    cvs.append(round(cross_val_score(modelrfc,X,y,cv=5).mean()*100,2))
    max_depths.append(estimator)
ax = sns.lineplot(x = max_depths, y = cvs)
ax.set(xlabel='Number of estimators', ylabel='Accuracy Score')
print(f'Best accuracy score |{max(cvs)}| Number of estimators: |{np.argmax(cvs) + 1}|')



# In[ ]:





# In[49]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
modelrffc=RandomForestClassifier(n_estimators=71)
modelrffc.fit(X_train,y_train)
yprd=modelrffc.predict(X_test)
cm=confusion_matrix(y_test, yprd)


# In[50]:


cm


# In[ ]:





# In[51]:


print(classification_report(y_test,yprd))


# In[ ]:





# In[52]:


dfa= df.drop(columns=['Duration of the credit in month',
                      'Credit amount in EUR',
                      'Age of the customer in years',
                      'Number of existing credits of the customer at this bank',
                      'Number of people the customer being liable to provide maintenance for',
                      'Present residence of the customer since (in years)',
                      'Installment rate in percentage of disposable income',
                       'label' ],axis=1)


# In[53]:


# There are 6715 cases where confident score is more than 0.8 which of 412 are exactly 1.
# Let's drop foreign worker column and Other guarantors since more than 80% of data are None.


# In[65]:



frq_items = apriori(fpm, min_support = 0.05, use_colnames = True) 
  
# Collecting the inferred rules in a dataframe 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules[rules['confidence']>0.9]


# In[70]:


rules.loc[3310]['antecedents']


# In[55]:


from scipy.stats import ttest_ind
dft


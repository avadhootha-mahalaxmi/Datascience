#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# In[4]:


orders_df=pd.read_csv("zomato.csv")
orders_df.head()


# In[6]:


import warnings
warnings.filterwarnings('ignore')


# In[7]:


orders_df.shape


# In[8]:


orders_df.columns.to_list()


# In[9]:


#Renaming columns
orders_df.rename(columns={'approx_cost(for two people)' : 'approx_cost','listed_in(city)' : 'area'}, inplace = True)


# In[10]:


#DATA CLEANING
#Dropping irrelevant columns
#here url,address,menu_item,reviews_list,dish_liked,phone,listed_in(type) are irrelavent for problem solution
orders_df.drop(['url','address','menu_item','reviews_list','dish_liked','phone','listed_in(type)'], axis=1 , inplace=True )
orders_df.head()


# In[11]:


#checking and handling the Datatypes
#Checking the types of data present in dataset
orders_df.info()


# In[12]:


#checking the unique values in rate column of dataset
orders_df['rate'].unique()


# In[13]:


#above result shows there are many null values and garbage values also
#replacing all null values and garbage values and making it to be converted into numbers
orders_df['rate'] = orders_df['rate'].str.replace("/5", " ")
orders_df['rate'] = orders_df['rate'].str.replace("nan", "NaN")
orders_df['rate'] = orders_df['rate'].str.replace("NEW" , "NaN")
orders_df['rate'] = orders_df['rate'].str.replace("-" ,"NaN")
orders_df['rate'] = orders_df['rate'].str.replace(" /5", "")
orders_df['rate'] = orders_df['rate'].fillna(np.nan)
orders_df['rate'] = orders_df['rate'].str.replace(" "," ")


# In[14]:


#verifying results
orders_df['rate'].unique()


# In[15]:


#changing datatype of rate column from object to float
orders_df['rate'] = orders_df['rate'].astype(float)


# In[16]:


#checking unique values in approx cost column of dataset
orders_df['approx_cost'].unique()


# In[17]:


#replacing null values and make it able to convert 
orders_df['approx_cost'] = orders_df['approx_cost'].str.replace("nan" , "NaN")
orders_df['approx_cost'] = orders_df['approx_cost'].fillna('NaN')
orders_df['approx_cost'] = orders_df['approx_cost'].str.replace("," , "")


# In[18]:


#verifying the results
orders_df['approx_cost'].unique()


# In[19]:


#changing datatype of column from object to float
orders_df['approx_cost'] = orders_df['approx_cost'].astype(float)


# In[20]:


orders_df.info()


# In[21]:


#checking the number of null values columnwise
orders_df.isna().sum()


# In[22]:


#checking of percentage of null values in each column
(orders_df.isna().sum() / orders_df.shape[0]) * 100


# In[23]:


#we dropping null values from columns having lesser number of null values
orders_df = orders_df[orders_df["location"].notna()]
orders_df = orders_df[orders_df["rest_type"].notna()]
orders_df = orders_df[orders_df["cuisines"].notna()]
orders_df = orders_df[orders_df["approx_cost"].notna()]

#verifying results
orders_df.isna().sum()


# In[24]:


#rate columns consists of huge null values but dropping this column leads to large amount of data.so,instead of dropping it ,we will impute with either mean,mode or median
#checking all statistics of rate column
orders_df['rate'].describe()


# In[25]:


#imputing null values with median of rate column
orders_df['rate'] = orders_df['rate'].fillna(orders_df['rate'].median())
orders_df.isna().sum()


# In[26]:


#now after dropping  values we will rest the index
orders_df.reset_index(inplace=True)
orders_df.drop(['index'],axis=1,inplace=True)


# ## Data visualization

# #### No.of orders vs Restaurants

# In[27]:


#checking for restaurants got higher orders
plt.figure(figsize= (11,4))
data = orders_df['name'].value_counts()[ : 30]
data.plot(kind='bar')
plt.xlabel('no of resturants',size= 14)
plt.ylabel('no of orders' , size = 14)
plt.title("Resturants with maximum no of orders", fontsize=15)
plt.show()


# ### Number of Restaurants having online order facility

# In[28]:


#checking no of resturants having online order facility
data = orders_df[['name' , 'online_order']].drop_duplicates()
plt.figure(figsize = (8,5))
ax= sns.countplot(x='online_order',data=data).set_title('online order facility',fontsize= 15)
plt.show()


# ### Number of restaurants having prebooking Table facility

# In[29]:


#checking number of restaurants having prebooking table facility
data = orders_df[['name' ,'book_table']].drop_duplicates()
plt.figure(figsize = (8,6))
ax = sns.countplot(x="book_table",data=data).set_title('pre booking facility', fontsize =15)
plt.show()


# ### Most common Ratings for orders

# In[30]:


#checking most common ratings for orders
data = orders_df.rate.value_counts().reset_index()[0:20]
sns.barplot(x= data['rate'] , y= data['count'])
plt.xlabel('Ratings')
plt.ylabel('no.of oders')
plt.title('most common ratings',size =15)
plt.show()


# #### we can see maximum number of orders got the 3.7 rating for restaurants

# ## Encoding columns

# In[31]:


#online order column
orders_df['online_order']= orders_df['online_order'].replace({"Yes": 1 ,"No" :0})
#Book_table column
orders_df['book_table']= orders_df['book_table'].replace({"Yes": 1 ,"No" :0})
orders_df.head()


# In[32]:


#label encode the categorical variables to make it easier to build algorithm

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[33]:


orders_df.location = le.fit_transform(orders_df.location )
orders_df.rest_type = le.fit_transform(orders_df.rest_type )
orders_df.cuisines = le.fit_transform(orders_df.cuisines )
orders_df.area = le.fit_transform(orders_df.area )


# In[34]:


orders_df.head()


# ### Feature selection

# In[35]:


#input features
x=orders_df.iloc[ : ,[1,2,4,5,6,7,8,9]]
x.head()


# In[36]:


#output feature
y=orders_df['rate']
y


# In[37]:


#splitting dataset into testing and training
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=0)


# ##### Importing ML models 

# In[69]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor


# ######   Linear Regression

# In[39]:


lr_model = LinearRegression()
lr_model.fit(x_train,y_train)


# In[40]:


y_lr= lr_model.predict(x_test)
y_lr[ :50]


# In[41]:


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score


# In[42]:


#metrics for linear regression

mae1 = mean_absolute_error(y_test,y_lr)
mse1 = mean_squared_error(y_test,y_lr)
r2score1 =r2_score(y_test,y_lr)
print("mean absolute error:",mae1)
print("mean squared error:" ,mse1)
print("Accuracy of model")
print("r2 score:" ,r2score1)


# #### Decision Tree Regressor

# In[43]:


d_tree = DecisionTreeRegressor()
d_tree.fit(x_train,y_train)


# In[44]:


y_dt= d_tree.predict(x_test)
y_dt[ :50]


# In[45]:


#metrics for DecisionTreeregressor

mae2 = mean_absolute_error(y_test,y_dt)
mse2 = mean_squared_error(y_test,y_dt)
r2score2 =r2_score(y_test,y_dt)
print("mean absolute error:",mae2)
print("mean squared error:" ,mse2)
print("Accuracy of model")
print("r2 score:" ,r2score2)


# #### Random Forest 

# In[63]:


rf=RandomForestRegressor(n_estimators=35 ,random_state=0)
rf.fit(x_train,y_train)


# In[64]:


y_rf = rf.predict(x_test)
y_rf[ :50]


# In[65]:


#metrics for random forest regressor
mae3 = mean_absolute_error(y_test,y_rf)
mse3= mean_squared_error(y_test,y_rf)
r2score3 =r2_score(y_test,y_rf)
print("mean absolute error:",mae3)
print("mean squared error:" ,mse3)
print("Accuracy of model")
print("r2 score:" ,r2score3)


# In[67]:


'''
for n in range(25,35):
    rc=RandomForestRegressor(n_estimators = n,random_state=0)
    rc.fit(x_train,y_train)
    y=rc.predict(x_test)
    r2score3 =r2_score(y_test,y)
    print(r2score3)
#At n_estimators=35 R2score=0.89567   
'''

# ### Extra Tree Regressor

# In[71]:


Et = ExtraTreesRegressor(n_estimators=120)
Et.fit(x_train,y_train)


# In[73]:


y_pred = Et.predict(x_test)
y_pred[:50]


# In[74]:


#metrics for Extra tree regressor
mae4 = mean_absolute_error(y_test,y_pred)
mse4= mean_squared_error(y_test,y_pred)
r2score4 =r2_score(y_test,y_pred)
print("mean absolute error:",mae4)
print("mean squared error:" ,mse4)
print("Accuracy of model")
print("r2 score:" ,r2score4)


# In[1]:


# use model by using pickle
import pickle
with open('model_pickle','wb') as f:
    pickle.dump(Et,f)


# In[ ]:





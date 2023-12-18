#!/usr/bin/env python
# coding: utf-8

# # Lead Scoring Case Study

# ### Step 1: Importing Libraries and Data

# #### 1.1 Import Libraries

# In[1]:


#import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#supress warnings
import warnings
warnings.filterwarnings("ignore")

# Sklearn libraries

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

#statmodel libraries
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


# In[2]:


#Environment settings

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option('display.width',None)

# pd.set_option('display.float_format', lambda x: '%.4f' % x) 


# ### 1.2 Reading the Data

# In[3]:


# Reading the dataset
df_leads = pd.read_csv("Leads.csv")


# In[4]:


df_leads.head()


# In[5]:


df_leads.shape


# The original dataframe has 9240 rows and 37 columns

# ### Step 2: Understanding and Inspecting the data 

# In[6]:


# Check summary of all numerical columns to understand the dataset better in terms of spread and 
# also spot anything unusual

df_leads.describe()


# In[7]:


# checking number of unique values in each column
df_leads.nunique()


# In[8]:


# Check data types of columns and nulls
df_leads.info()


#  There are **`null values`** in the dataframe

# In[9]:


#Checking for count of missing values in each column
df_leads.isnull().sum()


# NOTE: There are few columns with quite a high number of missing/null values in the dataframe. We will have to decide how to address them in data cleaning / data imputation step

# In[10]:


# Check for duplicate rows
print(df_leads.duplicated().sum())


# There are no duplicates in the dataframe df_leads

# ### Step 3: Data Cleaning

# ### 3.1 Treatment for 'Select' values

# As mentioned in the Problem Statement, many of the categorical variables have a level called **`'Select'`** **which needs to be handled because it is as good as a null value.** One of the reason might be is that the customer did not select any option from the list and hence for such columns the data remained as default 'Select' for Select. 

# In[11]:


# List of columns having 'Select' as value

cols_with_select_value = [col for col in df_leads.columns if len(df_leads[col].isin(['Select']).unique())>1]
print(cols_with_select_value)


# In[12]:


# Converting 'Select' values to NaN.
df_leads = df_leads.replace('Select', np.nan)


# In[13]:


# Checking if all 'Select' values have been handled in the columns
cols_with_select_value = [col for col in df_leads.columns if len(df_leads[col].isin(['Select']).unique())>1]
print(cols_with_select_value)


# Now, there are no `'Select'` values in the dataframe df_leads 

# ### 3.2 Handling Missing Values

# In[14]:


# Calculating Missing Values Percentage

100*(df_leads.isna().mean()).sort_values(ascending=False)


# #### 3.2.1 Drop Columns with more than 40% Null Values

# In[15]:


# user defined function to drop columns and know the shape before and after dropping

def dropNullColumns(data ,percentage=40):
    
    missing_perc = 100*(data.isna().mean()).sort_values(ascending=False)
    col_to_drop = missing_perc[missing_perc>=percentage].index.to_list()
    print("Total columns dropped: ",len(col_to_drop),"\n")
    print("List of columns dropped : " , col_to_drop,"\n")
    print("Shape before dropping columns: ",data.shape)
    
    data.drop(labels=col_to_drop,axis=1, inplace=True)
    
    print("Shape after dropping columns: ",data.shape)


# In[16]:


# dropping columns using UDF
dropNullColumns(df_leads) 


# In[17]:


# Checking the percentage of null values for remaining columns

100*(df_leads.isna().mean()).sort_values(ascending=False)


# ### 3.2.2 Columns with Categorical Data

# In[18]:


# Select the columns with non-numeric data type
categorical_cols = df_leads.select_dtypes(include=['category', 'object']).columns.tolist()

# Print the selected columns
print(categorical_cols)


# Approach would be to check the count of values in each categorical column and then decide how to treat the missing values for that particular column

# In[19]:


columnsList= ["City","Specialization","Tags",'What matters most to you in choosing a course',
              'What is your current occupation','Country','Last Activity','Lead Source']

for i in columnsList:
        perc=100*df_leads[i].value_counts(normalize=True)
        print("value_counts % for :",i,"\n")
        print(perc,"\n")
        print("___"*40,"\n")


# #### Insights:
# - City: City has 39.71 % missing values. Imputing missing values with Mumbai will make the data more skewed. Skewness will later cause bias in the model. Hence `City column can be dropped`. 
# 
# - Specialization: Specialization has 36.58 % missing values. The specialization selected is evenly distributed. Hence imputation or dropping is not a good choice. We need to create additional category called `'Others'`. 
# 
# - Tags has 36.29 % missing values. Tags are assigned to customers indicating the current status of the lead. Since this is current status, this column will `not be useful for modeling`. Hence it can be `dropped`.
# 
# - What matters most to you in choosing a course: This variable has 29.32 % missing values. 99.95% customers have selected 'better career prospects'. This is massively skewed and will `not provide any insight`.
# 
# - What is your current occupation: We can impute the missing values with `'Unemployed'` as it has the most values. This seems to be a important variable from business context, since X Education sells online courses and unemployed people might take this course to increase their chances of getting employed. 
# 
# - Country: X Education sells online courses and appx 96% of the customers are from India. Does not make business sense right now to impute missing values with India. Hence `Country column can be dropped.
# 
# - style= Last Activity:`"Email Opened"` is having highest number of values and overall missing values in this column is just 1.11%, hence we will impute the missing values with label `'Email Opened'`.
# 
# - Lead Source: `"Google"` is having highest number of occurences and overall nulls in this column is just 0.39%, hence we will impute the missing values with label 'Google'
# 

# Dropping the following columns
# - 'City',
# - 'Tags',
# - 'Country',
# - 'What matters most to you in choosing a course'

# In[20]:


# Dropping Columns
print("Before Drop",df_leads.shape)
df_leads.drop(['City','Tags','Country','What matters most to you in choosing a course'],axis=1,inplace=True)
print("After Drop",df_leads.shape)


# Imputing the following columns 
# - 'Specialization',
# - 'Lead Source',
# - 'Last Activity', 
# - 'What is your current occupation'

# In[21]:


# Imputing values as per the above observations/insights

missing_values={'Specialization':'Others','Lead Source':'Google','Last Activity':'Email Opened',
               'What is your current occupation':'Unemployed'}
df_leads=df_leads.fillna(value=missing_values)


# In[22]:


# Re Checking the percentage of null values for remaining columns

round(((df_leads.isnull().sum()/df_leads.shape[0])*100),2).sort_values(ascending=False)


# ### 3.2.3 Columns with Numerical Data

# In[23]:


# TotalVisits
print("TotalVisits - Value Counts")
print("----------------------------------------")
df_leads.TotalVisits.value_counts().head(10)


# Missing values in 'TotalVisits' can be imputed with `mode`.

# In[24]:


# TotalVisits missing values to be imputed with mode
df_leads['TotalVisits'].fillna(df_leads['TotalVisits'].mode()[0], inplace=True)


# In[25]:


# Page Views Per Visit
print("Page Views Per Visit - Value Counts")
print("----------------------------------------")
df_leads.TotalVisits.value_counts().head(10)


# Missing values in 'Page Views Per Visit' can be imputed with `mode`.

# In[26]:


# Page Views Per Visit missing values to be imputed with mode

df_leads['Page Views Per Visit'].fillna(df_leads['Page Views Per Visit'].mode()[0], inplace=True)


# <strong><span style="color:black">Re-checking the null values for columns </span></strong>

# In[27]:


# Re Checking the percentage of null values after handling categorical and numerical columns

round(((df_leads.isnull().sum()/df_leads.shape[0])*100),2).sort_values(ascending=False)


# ### 3.3 Removing Unwanted Columns

# In[28]:


# Last Notable Activity
print("Last Notable Activity")
print("----------------------------------------")
100*df_leads['Last Notable Activity'].value_counts(normalize=True)


# #### 3.3.1 Handling columns with only one unique value 

# In[29]:


#check for columns with one unique value, count and freq is same

df_leads.describe(include = 'object')


# Following columns have only **`one unique value`**: 
# - 'I agree to pay the amount through cheque', 
# - 'Get updates on DM Content', 
# - 'Update me on Supply Chain Content', 
# - 'Receive More Updates About Our Courses', 
# - 'Magazine'
# 
# These columns are of no use as they have <u>_only one category of response_</u> from customer and can be dropped:
# 

# In[30]:


# List of columns with one unique value whose count and frequency are same, we will drop these columns
cols_to_drop = ['Magazine','Receive More Updates About Our Courses',
                    'Update me on Supply Chain Content',
                    'Get updates on DM Content',
                    'I agree to pay the amount through cheque']

print("Before Dropping Columns",df_leads.shape)
df_leads.drop(cols_to_drop, axis = 1, inplace = True)
print("After Dropping Columns",df_leads.shape)


# #### 3.3.2 Dropping columns of no use for modeling

# NOTE: Columns such as:
#  
# - 'Prospect ID',
# - 'Lead Number',
# - 'Last Notable Activity' 
#  
# Above columns do not add any value to the model. Dropping these columns will remove unnecessary data from the dataframe.

# In[31]:


# Dropping Columns
print("Before Dropping Columns",df_leads.shape)
df_leads.drop(['Prospect ID','Lead Number','Last Notable Activity'],axis=1,inplace=True)
print("After Dropping Columns",df_leads.shape)


# In[32]:


# get the percentage of missing values in each row,output in descending order so high value will come on top

100*(df_leads.isna().mean(axis=1)).sort_values(ascending=False).head(10)


# No missing values in rows

# ### 3.4 Checking & Dropping Category Columns that are Skewed
# - This is similar to the handling of unique values in numeric columns

# #### 3.4.1 Checking skewness in categorical columns

# In[33]:


# plotting countplot for object dtype and histogram for number to get data distribution
categorical_col = df_leads.select_dtypes(include=['category', 'object']).columns.tolist()
plt.figure(figsize=(12,40))

plt.subplots_adjust(wspace=.2,hspace=2)
for i in enumerate(categorical_col):
    plt.subplot(8,2, i[0]+1)
    ax=sns.countplot(x=i[1],data=df_leads) 
    plt.xticks(rotation=90)
    
    for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')


plt.show()


# Following columns have data which is `highly skewed` :
# - 'Do Not Call',
# - 'Search', 
# - 'Newspaper Article',
# - 'X Education Forums',
# - 'Newspaper', 
# - 'Digital Advertisement',
# - 'Through Recommendations'. 
# 
# Hence these columns will be `dropped` as they will not add any value to the model. Morever, Skewed variables can affect the performance of logistic regression models, as they can `lead to biased` or `inaccurate parameter estimates`.

# In[34]:


# Dropping categorical columns with highly skewed data

print("Before Drop: ",df_leads.shape)
df_leads.drop(['Do Not Call','Search','Newspaper Article','X Education Forums','Newspaper','Digital Advertisement','Through Recommendations'],axis=1,inplace=True)
print("After Drop: ",df_leads.shape)


# In[35]:


df_leads.head()


# #### Data is clean from missing values 
#  - After data is cleaned lets standardise values 

# ### 3.5 Outlier Analysis

# #### For Numerical Columns

# In[36]:


def Check_Outliers(data,columnList):

    plt.figure(figsize=[22,11])
    plt.subplots_adjust(wspace=0.4,hspace=0.5)

    for i,j in enumerate(columnList):
        plt.subplot(2,2,i+1)

        sns.boxplot(y=data[j])     # y = df_leads[j] to make plot verticle

        plt.suptitle("\nChecking Outliers using Boxplot",fontsize=20,color="green")
        plt.ylabel(None)
        plt.title(j,fontsize=15,color='brown')


# In[37]:


# Checking outliers for numerical variables other than target variable 
num_cols = ["TotalVisits","Page Views Per Visit","Total Time Spent on Website"]

# UDF 
Check_Outliers(df_leads,num_cols)


# `"TotalVisits"`,`"Page Views Per Visit"`:Both these variables contain outliers as can be seen in the boxplot
# So, These outliers needs to be treated for these variables

# #### Capping Outliers for the treatment

# In[38]:


# before outlier treatment
df_leads.describe(percentiles=[.10,.25,.50,.75,.95])


# In[39]:


# Defining UDF to treat outliers via capping and flooring

def Outlier_treatment(df,columnList):
    for i in columnList:
        q1 = df[i].describe()["25%"]
        q3 = df[i].describe()["75%"]
        IQR = q3 - q1

        upper_bound = q3 + 1.5*IQR
        lower_bound = q1 - 1.5*IQR

        # capping upper_bound
        df[i] = np.where(df[i] > upper_bound, upper_bound,df[i])

        # flooring lower_bound
        df[i] = np.where(df[i] < lower_bound, lower_bound,df[i])
        
        


# In[40]:


# Checking outliers for numerical variables other than target variable 
capping_cols = ["TotalVisits","Page Views Per Visit"]

# UDF 
Outlier_treatment(df_leads,capping_cols)


# In[41]:


# Checking Boxplot after Outlier Treatment

num_cols = ["TotalVisits","Page Views Per Visit","Total Time Spent on Website"]

# UDF for boxplot
Check_Outliers(df_leads,num_cols)


# In[42]:


# after outlier treatment detailed percentile values
df_leads.describe(percentiles=[.10,.25,.50,.75,.95])


# ### 3.6 Fixing Invalid values & Standardising Data in columns
# 
# - Checking if entries in data are in correct format or not , casing styles (UPPER,lower)
# - Checking data types of columns

# In[43]:


df_leads.head()


# In[44]:


## Categorical Variables 

columnsList_cat = ["Lead Origin","Lead Source","Do Not Email","Last Activity","Specialization",
                  "What is your current occupation","A free copy of Mastering The Interview"]

for i in columnsList_cat:
        perc=100*df_leads[i].value_counts(normalize=True)
        print("value_counts % for :",i,"\n")
        print(perc,"\n")
        print("_^_"*40,"\n")


# 
# 
# - We've noticed that some categories/levels in the `"Lead Score" and "Last Activity"` columns have very few records. To prevent ending up with a bunch of unnecessary columns when we create dummy variables, we're planning to group these categories together under "Others". That way, we can keep things neat and tidy.
# 
# - Also we can see `"Google"` & `"google"` are same in `"Lead Source"`, so we will standardise the case.

# #### 3.6.1 Grouping Low frequency values

# In[45]:


# Grouping low frequency value levels to Others
df_leads['Lead Source'] = df_leads['Lead Source'].replace(["bing","Click2call","Press_Release",
                                                           "Social Media","Live Chat","youtubechannel",
                                                           "testone","Pay per Click Ads","welearnblog_Home",
                                                           "WeLearn","blog","NC_EDM"],"Others")

# Changing google to Google
df_leads['Lead Source'] = df_leads['Lead Source'].replace("google","Google")


# In[46]:


# value_counts percentage after replace
df_leads["Lead Source"].value_counts(normalize=True)*100


# In[47]:


# Grouping low frequency value levels to Others 
df_leads['Last Activity'] = df_leads['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                               'Had a Phone Conversation', 
                                                               'Approached upfront',
                                                               'View in browser link Clicked',       
                                                               'Email Marked Spam',                  
                                                               'Email Received','Visited Booth in Tradeshow',
                                                               'Resubscribed to emails'],'Others')


# In[48]:


# value_counts percentage after replace
df_leads['Last Activity'].value_counts(normalize=True)*100


# In[49]:


# Renaming column name to "Free_copy" from "A free copy of Mastering The Interview"
df_leads.rename(columns={'A free copy of Mastering The Interview': 'Free_copy'}, inplace=True)

# Renaming column name to "Current_occupation" from "What is your current occupationA free copy of Mastering The Interview"
df_leads.rename(columns={'What is your current occupation': 'Current_occupation'}, inplace=True)


# 
# - "Do Not Email" & "Free_copy" both are binary categorical columns lets map both of them yes/no to 1/0

# #### 3.6.2 Mapping Binary categorical variables

# In[50]:


# Mapping binary categorical variables (Yes/No to 1/0) 
df_leads['Do Not Email'] = df_leads['Do Not Email'].apply(lambda x: 1 if x =='Yes' else 0)

df_leads['Free_copy'] = df_leads['Free_copy'].apply(lambda x: 1 if x =='Yes' else 0)


# #### 3.6.3 Checking Data-types of variables

# In[51]:


df_leads.info()


# The data types appear to be suitable and no modifications are necessary.

# ## Step 4: Data Analysis (EDA)

# ### 4.1 Checking if Data is Imbalanced or not
# - Data is imbalance when one value is present in majority and other is in minority meaning an uneven distribution of observations in dataset
# - Data imbalance is in the context of Target variable only
# - `Target variable` is `'Converted'` which tells whether a past lead was converted or not wherein 1 means it was converted and 0 means it wasn’t converted 

# In[52]:


## ploting the results on bar plot

ax=(100*df_leads["Converted"].value_counts(normalize=True)).plot.bar(color=["Green","Red"],alpha=0.4)

# Adding and formatting title
plt.title("Leads Converted\n", fontdict={'fontsize': 16, 'fontweight' : 12, 'color' : 'Green'})


# Labeling Axes
plt.xlabel('Converted', fontdict={'fontsize': 12, 'fontweight' : 20, 'color' : 'Brown'})
plt.ylabel("Percentage Count", fontdict={'fontsize': 12, 'fontweight' : 20, 'color' : 'Brown'})

# modification ticks y axis
ticks=np.arange(0,101,20)
labels=["{:.0f}%".format(i) for i in ticks] 
plt.yticks(ticks,labels)

#xticks
plt.xticks([0,1],["No","Yes"])
plt.xticks(rotation=0)

for p in ax.patches:
    ax.annotate('{:.1f}%'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
                  ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    
plt.show()


# Insights:
# - **Conversion rate is of 38.5%**, meaning only 38.5% of the people have converted to leads.(Minority)
# - While 61.5% of the people didnt convert to leads. (Majority)

# In[53]:


### Ratio of Data Imbalance
ratio=(df_leads["Converted"].value_counts(normalize=True).loc[0])/(df_leads["Converted"].value_counts(normalize=True).loc[1])

print("Data Imbalance Ratio : {:.2f} : {}".format(ratio,1))


# ### 4.2 Univariate Analysis

# In[54]:


df_leads.head()


# #### 4.2.1 Univariate Analysis for Categorical Variables

# In[55]:


#List of categorical columns
cat_cols = ["Lead Origin","Current_occupation","Do Not Email",
            "Free_copy","Lead Source","Last Activity","Specialization"]


# In[56]:


# countplot of columns with its value_counts percentage as annotation
for i in cat_cols[:4]:
    
    plt.figure(figsize=[10,5])
    plt.title("Count plot of {}".format(i),color="green")
    ax=sns.countplot(x=i,data=df_leads)
    total=len(df_leads[i])
    plt.xticks(rotation=0)
    
    for p in ax.patches:
        text = '{:.1f}%'.format(100*p.get_height()/total)
        x = p.get_x() + p.get_width() / 2.
        y = p.get_height()
        
        ax.annotate(text, (x,y), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
        


# In[57]:


# Barplot for remaining columns from cat_cols (Did separate to rotate xticks 90* so labels doesnt become messy)
for i in cat_cols[4:]:
    
    plt.figure(figsize=[10,5])
    plt.title("Count plot of {}".format(i),color="green")
    ax=sns.countplot(x=i,data=df_leads)
    total=len(df_leads[i])
    plt.xticks(rotation=90)
    
    
    if i!="Specialization":        # (not doing for Specialization xtick labels will be messy)
        for p in ax.patches:
            text = '{:.1f}%'.format(100*p.get_height()/total)
            x = p.get_x() + p.get_width() / 2.
            y = p.get_height()

            ax.annotate(text, (x,y), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    else:
        pass


# Observations: 
# 
# - In Categorical Univariate Analysis we get to know the value counts percentage in each variable that how much is the distribution of values in each column.
# 
# - With this we get some understanding that which variables can be used in **Bivariate analysis**.
# 
# Insights Univariate: 
# 
# **Here is the list of features from variables which are present in majority (Converted and Not Converted included)** 
# 
# - **Lead Origin:** "Landing Page Submission" identified 53% customers, "API" identified 39%. 
# 
# - **Current_occupation:** It has 90% of the customers as Unemployed
# 
# - **Do Not Email:** 92% of the people has opted that they dont want to be emailed about the course.
# 
# - **Lead Source:** 58% Lead source is from Google & Direct Traffic combined
# 
# - **Last Activity:** 68% of customers contribution in SMS Sent & Email Opened activities
# 
# NOTE: These insights will be helpful in further Bivariate Analysis.

# ### 4.3 Bivariate Analysis
# - Bivariate analysis happens between two variables

# #### 4.3.1 Bivariate Analysis for Categorical Variables

# In[58]:


# UDF "Bivariate_cat" tells comparision between Actual Distribution (value_count percentage) from the data and 
# 2nd graph tell the Lead Conversion Rate in percentage (how much leads are converted from 1st plot distribution)

def Bivariate_cat(df,variable_name,Target="Converted"):
    plt.figure(figsize=(20,6))
    plt.suptitle("{} Countplot vs Lead Conversion Rates".format(variable_name),color="Brown", fontsize=18)
    
    # 1st plot in subplot
    plt.subplot(1,2,1)
    plt.title("Distribution of {}".format(variable_name),color="blue")
    ax=sns.countplot(x=variable_name,hue=Target,data=df_leads,palette="prism_r",alpha=0.46)
    
    total=len(df_leads[variable_name])
    plt.xticks(rotation=90)
    plt.legend(["No","Yes"],title = "Converted")
    
    # Annotation for 1st plot        
    for p in ax.patches:
        text = '{:.1f}%'.format(100*p.get_height()/total)
        x = p.get_x() + p.get_width() / 2.
        y = p.get_height()

        ax.annotate(text, (x,y), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    # 2nd plot
    plt.subplot(1,2,2)
    plt.title("Lead Conversion Rate of {}".format(variable_name),color="green",fontsize=12)
    ax=sns.countplot(x=variable_name,hue=Target,data=df,palette="BuGn",alpha=0.85)   #ax1 is for annotation
    
    # Modifications
    plt.xticks(rotation=90)
    plt.ylabel("Count",color='brown')
    plt.xlabel("{}".format(variable_name))
    plt.legend(labels=["Not Converted","Converted"],title = "Lead Conversion Rate")
    
    # Annotation for 2nd plot
    # Calculate percentage above bars in countplot (Conversion rate)
    all_heights = [[p.get_height() for p in bars] for bars in ax.containers]
    for bars in ax.containers:
        for i, p in enumerate(bars):
            total = sum(xgroup[i] for xgroup in all_heights)
            percentage = f'{(100 * p.get_height() / total) :.1f}%'
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height()), size=11, ha='center', va='bottom')


# In[59]:


# Bivariate Analysis for all these variables using loop and UDF
# Comparision between variables w.r.t. 'Converted' (Target variable) , taking one categorical column w.r.t target variable as 'hue'
cat_cols = ["Lead Origin","Current_occupation","Do Not Email",
            "Lead Source","Last Activity","Specialization","Free_copy"]

for i in cat_cols:
    Bivariate_cat(df_leads,variable_name=i)
    


# Insights:
# 
# - Lead Origin: Around 52% of all leads originated from _"Landing Page Submission"_ with a **lead conversion rate (LCR) of 36%**.The _"API"_ identified approximately 39% of customers with a **lead conversion rate (LCR) of 31%**.
# 
# - Current_occupation: Around 90% of the customers are _Unemployed_ with **lead conversion rate (LCR) of 34%**. While _Working Professional_ contribute only 7.6% of total customers with almost **92% lead conversion rate (LCR)**.
# 
# - Do Not Email: 92% of the people has opted that they dont want to be emailed about the course. 
# 
# Note: We have assumed **LCR** as **Lead Conversion Rate** in short form.
# 
# 
# - Lead Source: Google_ has **LCR of 40%** out of 31% customers , Direct Traffic_ contributes **32% LCR** with 27% customers which is lower than Google,Organic Search_ also gives **37.8% of LCR** but the contribution is by only 12.5% of customers ,Reference_ has **LCR of 91%** but there are only around 6% of customers through this Lead Source.
# 
# - Last Activity:'SMS Sent'_ has **high lead conversion rate of 63%** with 30% contribution from last activities, 'Email Opened'_ activity contributed 38% of last activities performed by the customers with 37% lead conversion rate.
# 
# - Specialization: Marketing Managemt,HR Management,Finance Management shows good contribution.

# #### 4.3.2 Bivariate Analysis for Numerical Variables

# In[60]:


plt.figure(figsize=(16, 4))
sns.pairplot(data=df_leads,vars=num_cols,hue="Converted")                                  
plt.show()


# In[61]:


num_cols =["Converted",'TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']


# In[62]:


# Heatmap to show correlation between numerical variables
sns.heatmap(data=df_leads[num_cols].corr(),cmap="Blues",annot=True)
plt.show()


# In[63]:


# Boxplot with Converted as hue

plt.figure(figsize=(15, 5))
plt.subplot(1,3,1)
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = df_leads)
plt.subplot(1,3,2)
sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data = df_leads)
plt.subplot(1,3,3)
sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = df_leads)
plt.show()


# Insights:
# - Past Leads who spends more time on Website are successfully converted than those who spends less as seen in the boxplot

# ## Step 5: Data Preparation

# ### 5.1 Dummy Variables
# - For categorical variables with multiple levels, create dummy features (one-hot encoded)

# #### Binary level categorical columns are already mapped to 1 / 0 in previous steps, So start with Dummy variable creation

# In[64]:


df_leads.head()


# In[65]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy = pd.get_dummies(df_leads[["Lead Origin","Lead Source","Last Activity","Specialization","Current_occupation"]], drop_first=True)

# Adding the results to the master dataframe
df_leads = pd.concat([df_leads, dummy], axis=1)


# In[66]:


df_leads.head()


# In[67]:


# We have created dummies for the below variables, so we can drop them

df_leads = df_leads.drop(["Lead Origin","Lead Source","Last Activity","Specialization","Current_occupation"],1)


# In[68]:


df_leads.shape


# In[69]:


df_leads.info()


# ## Step 6: Test-Train Split 

# In[70]:


# Putting predictor variables to X
X = df_leads.drop('Converted', axis=1)

# Putting Target variables to y
y = df_leads["Converted"]


# In[71]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[72]:


print("X_train:", X_train.shape,"\ny_train:",y_train.shape)


# In[73]:


print("X_test:", X_test.shape,"\ny_test:",y_test.shape)


# ## Step 7: Feature Scaling 

# In[74]:


# using standard scaler for scaling the features
scaler = StandardScaler()

# fetching int64 and float64 dtype columns from dataframe for scaling
num_cols=X_train.select_dtypes(include=['int64','float64']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])


# In[75]:


# X-train dataframe after standard scaling
X_train.head()


# In[76]:


# Checking the Lead Conversion Rate (LCR) - "Converted" is our Target Variable
# We will denote Lead Conversion Rate with 'LCR' as its short form

LCR = (sum(df_leads['Converted'])/len(df_leads['Converted'].index))*100
LCR


# <strong><span style="color:Blue">Note:</span></strong> We have 38.5% Conversion Rate

# ### 7.1 : Looking at Correlations
# - Feature elimination based on correlations

# In[77]:


# analyse correlation matrix
plt.figure(figsize = (20,15))        
sns.heatmap(df_leads.corr(),linewidths=0.01,cmap="Blues",annot=True)
plt.show()


# In[78]:


# as the above heatmap has so many columns lets breakdown suspected variables which migh have high correlation with each other
# analysing variables which might be highly correlated with each other from same class from above graph
plt.figure(figsize = (5,5))        
sns.heatmap(df_leads[["Lead Source_Facebook","Lead Origin_Lead Import","Lead Origin_Lead Add Form","Lead Source_Reference"]].corr(),linewidths=0.01,cmap="Blues",annot=True)
plt.show()


# Note: These predictor variables above are very highly correlated with each other near diagonal with (0.98 and 0.85), it is better that we drop one of these variables from each pair as they won’t add much value to the model. So , we can drop any of them, lets drop `'Lead Origin_Lead Import'` and `'Lead Origin_Lead Add Form'`.

# In[79]:


X_test = X_test.drop(['Lead Origin_Lead Import','Lead Origin_Lead Add Form'],1)

X_train = X_train.drop(['Lead Origin_Lead Import','Lead Origin_Lead Add Form'],1)


# ## Step 8: Model Building 
# - We will Build Logistic Regression Model for predicting categorical variable
# - Feature Selection Using RFE (Coarse tuning)
# - Manual fine-tuning using p-values and VIFs

# ### 8.1 Feature Selection Using RFE (Recursive Feature Elimination)

# In[80]:


# Lets use RFE to reduce variables 
logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=15)            
rfe = rfe.fit(X_train, y_train)


# In[81]:


#checking the output of RFE
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[82]:


# instead of using as zip using this for more readablility

# Check the top 15 features chosen by RFE
top15=pd.DataFrame()
top15['features']=X_train.columns
top15['Feature Chosen'] = rfe.support_
top15['Ranking']=rfe.ranking_
top15.sort_values(by='Ranking')


# In[83]:


# columns which are selected by RFE
rfe_col = X_train.columns[rfe.support_]
rfe_col


# In[84]:


# columns which are not selected by RFE
X_train.columns[~rfe.support_]


# In[85]:


# User defined function for calculating VIFs for variables
def get_vif(model_df):
    X = pd.DataFrame()
    X['Features'] = model_df.columns
    X['VIF'] = [variance_inflation_factor(model_df.values, i) for i in range(model_df.shape[1])]
    X['VIF'] = round(X['VIF'], 2)
    X = X.sort_values(by='VIF', ascending=False)
    X = X.reset_index(drop=True)
    return X


# ### Model 1

# In[86]:


# Building model using statsmodels, for the detailed statistics

# columns selected by RFE to be used for this model 
rfe_col=X_train.columns[rfe.support_]

# Creating X_train dataframe with variables selected by RFE
X_train_rfe = X_train[rfe_col]

# Adding a constant variable 
X_train_sm1 = sm.add_constant(X_train_rfe)

# Create a fitted model
logm1 = sm.GLM(y_train,X_train_sm1,family = sm.families.Binomial()).fit()  

logm1.params


# In[87]:


#Let's see the summary of our logistic regression model
print(logm1.summary())


# NOTE : "Current_occupation_Housewife" column will be removed from model due to high p-value of 0.999, which is above the accepted threshold of 0.05 for statistical significance.

# ### Model 2

# In[88]:


# Dropping 'Current_occupation_Housewife' column
rfe_col=rfe_col.drop("Current_occupation_Housewife")


# In[89]:


# Creating X_train dataframe with variables selected by RFE
X_train_rfe = X_train[rfe_col]

# Adding a constant variable 
X_train_sm2 = sm.add_constant(X_train_rfe)

# Create a fitted model
logm2 = sm.GLM(y_train,X_train_sm2,family = sm.families.Binomial()).fit()  

logm2.params


# In[90]:


#Let's see the summary of our logistic regression model
print(logm2.summary())


# NOTE: "Lead Source_Facebook" column will be removed from model due to high p-value of 0.187, which is above the accepted threshold of 0.05 for statistical significance.

# ### Model 3

# In[91]:


# Dropping 'Lead Source_Facebook' column
rfe_col=rfe_col.drop("Lead Source_Facebook")


# In[92]:


# Creating X_train dataframe with variables selected by RFE
X_train_rfe = X_train[rfe_col]

# Adding a constant variable 
X_train_sm3 = sm.add_constant(X_train_rfe)

# Create a fitted model
logm3 = sm.GLM(y_train,X_train_sm3,family = sm.families.Binomial()).fit()  

logm3.params


# In[93]:


#Let's see the summary of our logistic regression model
print(logm3.summary())


# NOTE: "Lead Source_Others" column will be removed from model due to high p-value of 0.055, which is above the accepted threshold of 0.05 for statistical significance.

# ### Model 4

# In[94]:


# Dropping 'Lead Source_Facebook' column
rfe_col=rfe_col.drop("Lead Source_Others")


# In[95]:


# Creating X_train dataframe with variables selected by RFE
X_train_rfe = X_train[rfe_col]

# Adding a constant variable 
X_train_sm4 = sm.add_constant(X_train_rfe)

# Create a fitted model
logm4 = sm.GLM(y_train,X_train_sm4,family = sm.families.Binomial()).fit()  

logm4.params


# In[96]:


#Let's see the summary of our logistic regression model
print(logm4.summary())


# NOTE: Model 4 is stable and has significant p-values within the threshold (p-values < 0.05), so we will use it for further analysis.
# 
# - Now lets check VIFs for these variables to check if there is any multicollinearity which exists among the independent variables

# In[97]:


# Now checking VIFs for all variables in the Model 4 
get_vif(X_train_rfe)


# NOTE: No variable needs to be dropped as they all have good VIF values less than 5.
# - p-values for all variables is less than 0.05
# - This model looks acceptable as everything is under control (p-values & VIFs).
# - So we will final our Model 4 for `Model Evaluation`.

# ## Step 9: Model Evaluation 
# - Confusion Matrix
# - Accuracy
# - Sensitivity and Specificity
# - Threshold determination using ROC & Finding Optimal cutoff point
# - Precision and Recall

# In[98]:


# Getting the predicted values on the train set
y_train_pred = logm4.predict(X_train_sm4)           # giving prob. of getting 1

y_train_pred[:10]


# In[99]:


# for array
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[100]:


# Creating a dataframe with the actual churn flag and the predicted probabilities

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()

# y_train.values actual Converted values from df_leads dataset
# y_train_pred probability of Converted values predicted by model


# NOTE: Now we have to find the optimal cutoff Threshold value of Probability. Let's start with default 0.5 value and add a new feature predicted in above dataframe using the probabilities

# In[101]:


y_train_pred_final['Predicted'] = y_train_pred_final["Converted_Prob"].map(lambda x: 1 if x > 0.5 else 0)

# checking head
y_train_pred_final.head()


# ### 9.1 Confusion Matrix

# In[102]:


# Confusion matrix  (Actual / predicted)

confusion = metrics.confusion_matrix(y_train_pred_final["Converted"], y_train_pred_final["Predicted"])
print(confusion)


# In[103]:


# Predicted        not_converted  |  converted
# Actual                          |
# -----------------------------------------------------
# not_converted       3588       |   414
# converted           846        |   1620  


# Above is the confusion matrix when we use threshold of probability as 0.5


# ### 9.2 Accuracy

# In[104]:


# Checking the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final["Converted"], y_train_pred_final["Predicted"]))


# ### 9.3 Metrics beyond simply accuracy
# - Sensitivity and Specificity
# - When we have Predicted at threshold 0.5 probability

# In[105]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[106]:


# Let's see the sensitivity of our logistic regression model
print("Sensitivity :",TP / float(TP+FN))


# In[107]:


# Let us calculate specificity
print("Specificity :",TN / float(TN+FP))


# In[108]:


# Calculate false postive rate - predicting conversion when customer does not have converted
print(FP/ float(TN+FP))


# In[109]:


# positive predictive value 
print (TP / float(TP+FP))


# In[110]:


# Negative predictive value
print (TN / float(TN+ FN))


# ### 9.4 Plotting the ROC Curve
# 
# An ROC curve demonstrates several things:
# 
# - It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
# - The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
# - The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

# In[111]:


# UDF to draw ROC curve 
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[112]:


fpr, tpr, thresholds = metrics.roc_curve(y_train_pred_final["Converted"], y_train_pred_final["Converted_Prob"], drop_intermediate = False )


# In[113]:


# Drawing ROC curve for Train Set
draw_roc(y_train_pred_final["Converted"], y_train_pred_final["Converted_Prob"])


# NOTE: Area under ROC curve is 0.88 out of 1 which indicates a good predictive model

# ### 9.4.1 Finding Optimal Cutoff Point/ Probability
# - It is that probability where we get `balanced sensitivity and specificity`

# In[114]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final['Converted_Prob'].map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[115]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final["Converted"], y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[116]:


# Let's plot accuracy sensitivity and specificity for various probabilities.

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

# Finding the intersection points of the sensitivity and accuracy curves
sensi_interp = interp1d(cutoff_df['prob'], cutoff_df['sensi'], kind='linear')
acc_interp = interp1d(cutoff_df['prob'], cutoff_df['accuracy'], kind='linear')
intersection_1 = np.round(float(fsolve(lambda x : sensi_interp(x) - acc_interp(x), 0.5)), 3)

# Find the intersection points of the specificity and accuracy curves
speci_interp = interp1d(cutoff_df['prob'], cutoff_df['speci'], kind='linear')
intersection_2 = np.round(float(fsolve(lambda x : speci_interp(x) - acc_interp(x), 0.5)), 3)

# Calculate the average of the two intersection points
intersection_x = (intersection_1 + intersection_2) / 2

# Interpolate the accuracy, sensitivity, and specificity at the intersection point
accuracy_at_intersection = np.round(float(acc_interp(intersection_x)), 2)
sensitivity_at_intersection = np.round(float(sensi_interp(intersection_x)), 2)
specificity_at_intersection = np.round(float(speci_interp(intersection_x)), 2)

# Plot the three curves and add vertical and horizontal lines at intersection point
cutoff_df.plot.line(x='prob', y=['accuracy', 'sensi', 'speci'])
plt.axvline(x=intersection_x, color='grey',linewidth=0.55, linestyle='--')
plt.axhline(y=accuracy_at_intersection, color='grey',linewidth=0.55, linestyle='--')

# Adding annotation to display the (x,y) intersection point coordinates 
plt.annotate(f'({intersection_x} , {accuracy_at_intersection})',
             xy=(intersection_x, accuracy_at_intersection),
             xytext=(0,20),
             textcoords='offset points',
             ha='center',
             fontsize=9)

# Displaying the plot
plt.show()


# NOTE: 0.345 is the approx. point where all the curves meet, so 0.345 seems to be our `Optimal cutoff point` for probability threshold .
# - Lets do mapping again using optimal cutoff point 

# In[117]:


y_train_pred_final['final_predicted'] = y_train_pred_final['Converted_Prob'].map( lambda x: 1 if x > 0.345 else 0)

# deleting the unwanted columns from dataframe
y_train_pred_final.drop([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,"Predicted"],axis = 1, inplace = True) 
y_train_pred_final.head()


# ### 9.5 Calculating all metrics using confusion matrix for Train

# In[118]:


# Checking the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final["Converted"], y_train_pred_final["final_predicted"]))

# or can be found using confusion matrix with formula, lets find all matrix in one go ahead using UDF


# In[119]:


# UDF for all Logistic Regression Metrics
def logreg_all_metrics(confusion_matrix):
    TN =confusion_matrix[0,0]
    TP =confusion_matrix[1,1]
    FP =confusion_matrix[0,1]
    FN =confusion_matrix[1,0]
    
    accuracy = (TN+TP)/(TN+TP+FN+FP)
    sensi = TP/(TP+FN)
    speci = TN/(TN+FP)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    TPR = TP/(TP + FN)
    TNR = TN/(TN + FP)
    
    #Calculate false postive rate - predicting conversion when customer does not have converted
    FPR = FP/(FP + TN)     
    FNR = FN/(FN +TP)
    
    print ("True Negative                    : ", TN)
    print ("True Positive                    : ", TP)
    print ("False Negative                   : ", FN)
    print ("False Positve                    : ", FP) 
    
    print ("Model Accuracy                   : ", round(accuracy,4))
    print ("Model Sensitivity                : ", round(sensi,4))
    print ("Model Specificity                : ", round(speci,4))
    print ("Model Precision                  : ", round(precision,4))
    print ("Model Recall                     : ", round(recall,4))
    print ("Model True Positive Rate (TPR)   : ", round(TPR,4))
    print ("Model False Positive Rate (FPR)  : ", round(FPR,4))
    
    


# In[120]:


# Finding Confusion metrics for 'y_train_pred_final' df
confusion_matrix = metrics.confusion_matrix(y_train_pred_final['Converted'], y_train_pred_final['final_predicted'])
print("*"*50,"\n")

#
print("Confusion Matrix")
print(confusion_matrix,"\n")

print("*"*50,"\n")

# Using UDF to calculate all metrices of logistic regression
logreg_all_metrics(confusion_matrix)

print("\n")
print("*"*50,"\n")


# ### 9.6 Precision and recall tradeoff
# - Let's compare all metrics of Precision-Recall view with Specificity-Sensivity view and get better probability threshold for boosting conversion rate to 80% as asked by CEO.

# In[121]:


# Creating precision-recall tradeoff curve
y_train_pred_final['Converted'], y_train_pred_final['final_predicted']
p, r, thresholds = precision_recall_curve(y_train_pred_final['Converted'], y_train_pred_final['Converted_Prob'])


# In[122]:


# plot precision-recall tradeoff curve
plt.plot(thresholds, p[:-1], "g-", label="Precision")
plt.plot(thresholds, r[:-1], "r-", label="Recall")

# add legend and axis labels

plt.axvline(x=0.41, color='teal',linewidth = 0.55, linestyle='--')
plt.legend(loc='lower left')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')

plt.show()


# NOTE : The intersection point of the curve is the threshold value where the model achieves a balance between precision and recall. It can be used to optimise the performance of the model based on business requirement,Here our probability threshold is 0.41 aprrox from above curve.

# In[123]:


# copying df to test model evaluation with precision recall threshold of 0.41
y_train_precision_recall = y_train_pred_final.copy()


# In[124]:


# assigning a feature for 0.41 cutoff from precision recall curve to see which one is best view (sensi-speci or precision-recall)
y_train_precision_recall['precision_recall_prediction'] = y_train_precision_recall['Converted_Prob'].map( lambda x: 1 if x > 0.41 else 0)
y_train_precision_recall.head()


# In[125]:


## Lets see all matrics at 0.41 cutoff in precision-recall view and compare it with 0.345 cutoff from sensi-speci view

# Finding Confusion metrics for 'y_train_precision_recall' df
confusion_matrix = metrics.confusion_matrix(y_train_precision_recall['Converted'], y_train_precision_recall['precision_recall_prediction'])
print("*"*50,"\n")

#
print("Confusion Matrix")
print(confusion_matrix,"\n")

print("*"*50,"\n")

# Using UDF to calculate all metrices of logistic regression
logreg_all_metrics(confusion_matrix)

print("\n")
print("*"*50,"\n")


# NOTE:
# - As we can see in above metrics when we used precision-recall threshold cut-off of 0.41 the values in True Positive Rate ,Sensitivity, Recall have dropped to around 75%, but we need it close to 80% as the Business Objective.
# - 80% for the metrics we are getting with the sensitivity-specificity cut-off threshold of 0.345. So, we will go with sensitivity-specificity view for our Optimal cut-off for final predictions.
# 

# ### Adding Lead Score Feature to Training dataframe  
# - A higher score would mean that the lead is hot, i.e. is most likely to convert 
# - Whereas a lower score would mean that the lead is cold and will mostly not get converted.

# In[126]:


# Lets add Lead Score 

y_train_pred_final['Lead_Score'] = y_train_pred_final['Converted_Prob'].map( lambda x: round(x*100))
y_train_pred_final.head()


# ## Step 10: Making Predictions on test set

# ### 10.1 Scaling Test dataset

# In[127]:


X_test.info()


# In[128]:


# fetching int64 and float64 dtype columns from dataframe for scaling
num_cols=X_test.select_dtypes(include=['int64','float64']).columns

# scaling columns
X_test[num_cols] = scaler.transform(X_test[num_cols])

X_test = X_test[rfe_col]
X_test.head()


# ### 10.2 Prediction on Test Dataset using final model 

# In[129]:


# Adding contant value
X_test_sm = sm.add_constant(X_test)
X_test_sm.shape


# In[130]:


# making prediction using model 4 (final model)
y_test_pred = logm4.predict(X_test_sm)


# In[131]:


# top 10 columns
y_test_pred[:10]


# In[132]:


# Changing to dataframe of predicted probability
y_test_pred = pd.DataFrame(y_test_pred)
y_test_pred.head()


# In[133]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
y_test_df.head()


# In[134]:


# Putting Prospect ID to index
y_test_df['Prospect ID'] = y_test_df.index

# Removing index for both dataframes to append them side by side 
y_test_pred.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

# Appending y_test_df and y_test_pred
y_pred_final = pd.concat([y_test_df, y_test_pred],axis=1)
y_pred_final.head()


# In[135]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Prob'})

# Rearranging the columns
y_pred_final = y_pred_final.reindex(['Prospect ID','Converted','Converted_Prob'], axis=1)

y_pred_final.head()


# In[136]:


# taking sensitivity-specificity method at 0.345 probability cutoff during training
y_pred_final['final_predicted'] = y_pred_final['Converted_Prob'].map(lambda x: 1 if x > 0.345 else 0)
y_pred_final.head()


# In[137]:


# Drawing ROC curve for Test Set
fpr, tpr, thresholds = metrics.roc_curve(y_pred_final["Converted"], y_pred_final["Converted_Prob"], drop_intermediate = False )

draw_roc(y_pred_final["Converted"], y_pred_final["Converted_Prob"])


# NOTE: Area under ROC curve is 0.87 out of 1 which indicates a good predictive model

# NOTE:
# - Now that the final predictions have been made, the next step would be to evaluate the performance of the predictive model on a test set. 
# - We will do this by comparing the predicted labels (final_predicted) to the actual labels (Converted) to compute various performance metrics such as accuracy, precision, recall, etc.

# ### 10.3 Test set Model Evaluation
# - Calculating all metrics using confusion matrix for Test set

# In[138]:


# Finding Confusion metrics for 'y_train_pred_final' df
confusion_matrix = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final['final_predicted'])
print("*"*50,"\n")

#
print("Confusion Matrix")
print(confusion_matrix,"\n")

print("*"*50,"\n")

# Using UDF to calculate all metrices of logistic regression
logreg_all_metrics(confusion_matrix)

print("\n")
print("*"*50,"\n")


# NOTE: The evaluation matrics are pretty close to each other so it indicates that the model is performing consistently across different evaluation metrics in both test and train dataset.
# - `For Test set`
#     - Accuracy : 80.34%
#     - Sensitivity : 79.82% ≈ 80%
#     - Specificity : 80.68%
#     
# These matrics are very close to train set, so out final model logm4 is  performing with good consistency on both Train & Test set

# In[139]:


# features and their coefficicent from final model
parameters=logm4.params.sort_values(ascending=False)
parameters


# NOTE: A high positive coefficient indicates that a variable has a stronger influence on predicting the probability of leads converting to take up X-Education's course.

# ## Adding Lead Score Feature to Test dataframe 
# - A higher score would mean that the lead is hot, i.e. is most likely to convert 
# - Whereas a lower score would mean that the lead is cold and will mostly not get converted.

# In[140]:


# Lets add Lead Score 

y_pred_final['Lead_Score'] = y_pred_final['Converted_Prob'].map( lambda x: round(x*100))
y_pred_final.head()


# <strong><span style="color:purple">Lead Score: </span></strong> Lead Score is assigned to the customers
# - The customers with a higher lead score have a higher conversion chance 
# - The customers with a lower lead score have a lower conversion chance.

# <hr/>

# # Conclusion  
# 
# ## Train - Test
# ### Train Data Set:         
# 
# - Accuracy: 80.46%
# 
# - Sensitivity: 80.05%
# 
# - Specificity: 80.71%
# 
# ### Test Data Set:
# 
# - Accuracy: 80.34%
# 
# - Sensitivity: 79.82% ≈ 80%
# 
# - Specificity: 80.68%
#  
# 
# NOTE: The evaluation matrics are pretty close to each other so it indicates that the model is performing consistently across different evaluation metrics in both test and train dataset.
# 
# - The model achieved a `sensitivity of 80.05%` in the train set and 79.82% in the test set, using a cut-off value of 0.345.
# - Sensitivity in this case indicates how many leads the model identify correctly out of all potential leads which are converting
# - `The CEO of X Education had set a target sensitivity of around 80%.`
# - The model also achieved an accuracy of 80.46%, which is in line with the study's objectives.
# <hr/>
# 
# 

# ## Model parameters
# - The final Logistic Regression Model has 12 features
# 
# ### `Top 3 features` that contributing `positively` to predicting hot leads in the model are:
# - Lead Source_Welingak Website
# 
# - Lead Source_Reference 
# 
# - Current_occupation_Working Professional
# 
# NOTE: The Optimal cutoff probability point is 0.345.Converted probability greater than 0.345 will be predicted as Converted lead (Hot lead) & probability smaller than 0.345 will be predicted as not Converted lead (Cold lead).

# # Recommendations  
# 
# ### To increase our Lead Conversion Rates: 
# 
# - Focus on features with positive coefficients for targeted marketing strategies.
# - Develop strategies to attract high-quality leads from top-performing lead sources.
# - Engage working professionals with tailored messaging.
# - Optimize communication channels based on lead engagement impact.
# - More budget/spend can be done on Welingak Website in terms of advertising, etc.
# - Incentives/discounts for providing reference that convert to lead, encourage providing more references.
# - Working professionals to be aggressively targeted as they have high conversion rate and will have better financial situation to pay higher fees too. 
# 
# 
# ### To identify areas of improvement:   
# 
# - Analyze negative coefficients in specialization offerings.
# - Review landing page submission process for areas of improvement.
# 
# 
# 
# 

# In[ ]:





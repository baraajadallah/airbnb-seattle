#!/usr/bin/env python
# coding: utf-8

# # Seattle Airbnb Data Analysis
# 
# > Since 2008, guests and hosts have used Airbnb to travel in a more unique, personalized way. As part of the Airbnb Inside initiative, this dataset describes the listing activity of homestays in Seattle, WA.
# 
# > In this notebook we will take a look at the Seattle Airbnb open data, analyze it and try to answer the following questions:
# 
# > 1- How does the number of listings differ between the different neighbourhoods in the city?
# 
# > 2- What are the most common amenities in general, and what are they in the most frequent neighbourhoods?
# 
# > 3- What are the properties that affects listing prices the most? In addition prices predictions of new listings.

# In the cells below, I will start getting to know the data and figure out how can I use it to answer my questions.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

main_df = pd.read_csv('listings.csv')


# In[2]:


main_df.head()


# In[3]:


no_nulls = set(main_df.columns[main_df.isnull().mean()==0])
no_nulls


# In[4]:


all_nulls = set(main_df.columns[main_df.isnull().mean() == 1])
all_nulls


# In[5]:


main_df = main_df.drop('license', axis=1)


# In the following steps, I wrnagle and clean the data as well as try to come up with a dataframe that most useful to answer the questions mentioned above, and as I go along with answering the questions, I will generate new data frames from the main one in a way suitable to asnwer individual questions.

# In[6]:


filtered_df = pd.DataFrame()
for col in main_df.columns:
    if main_df[col].nunique() < 30:
        filtered_df[col] = main_df[col]
 
filtered_df.columns


# In[7]:


# at first, make a data frame with most useful columns with limited number in unique entries.
cat_df1 = filtered_df[['host_identity_verified', 'room_type', 'bed_type', 'cancellation_policy','property_type']]


# In[8]:


# second, make a data frame with other useful categorical columns didn't appear above.
cat_df2 = main_df[['neighbourhood', 'amenities']]


# In[9]:


cat_df = pd.concat([cat_df1, cat_df2], axis=1)


# In[10]:


cat_df.isnull().any()


# In[11]:


fill_mode = lambda col: col.fillna(col.mode()[0])

cat_df.apply(fill_mode, axis=0)


# > Now I will exclude a data frame with numerical variables and concatenate both data frames together in useful_df

# In[12]:


main_df['price'] = (main_df['price'].str.extract(r'(\d+\.\d+|\d+)', expand=False).astype(float).round(2))


# In[13]:


main_df.info()


# In[14]:


num_df = main_df.select_dtypes(include=['float', 'int'])
num_df.head(10)


# In[15]:


'price' in num_df.columns


# In[16]:


num_df.columns


# In[17]:


num_df.columns[num_df.isnull().mean() > 0.8]


# In[18]:


num_df = num_df.drop('square_feet', axis=1)


# In[19]:


num_df.isnull().any()


# In[20]:


fill_mean = lambda col: col.fillna(col.mean())

num_df.apply(fill_mean, axis=0)


# In[21]:


useful_df = pd.concat([num_df, cat_df], axis=1)


# In[22]:


useful_df.head(10)


# In[23]:


useful_df['price'].isnull().any()


# ## Question 1: What are the neighbourhoods with the most listings?
# 
# > below, we will take a look at listings numbers in  different neighborhoods and what are the neighbouhoods with the most listings, the data set contains over 80 neighbourhoods which makes it messy if we want to observe the numbers and plot them for all neoghbourhoods, so we will take a look at the top 20 neighbourhoods listings numbers.
# 
# > And I will use the same neighbourhoods to answer the second question regarding the most common aminities.

# In[24]:


freq_n_hoods = useful_df['neighbourhood'].value_counts()[:20]
hoods_lst = freq_n_hoods.index.tolist()


# In[25]:


useful_df.neighbourhood.nunique()


# In[26]:


freq_n_hoods.plot(kind="bar");
plt.title("Listing Numbers per Neighbourhood")
plt.xlabel('Neighbourhood')
plt.ylabel('Number of Listings');


# From the above bar chart, we notice that some neighbourhoods are superior among others in the number of listings especially Capitol Hill, followed by Ballard, Beltown, Minor, and Quenn Anne neighbourhoods.

# In[27]:


# Cleaning the amenities column in the useful data frame to make answering the second question easier
useful_df['amenities'] = useful_df.amenities.str.replace("[{}]", "").str.replace('"', "")
useful_df['amenities'].head()


# ## Question 2: What are the most common amenities in general, and what are they in the most frequent neighbourhoods?
# 
# > At first. I will take a look at the most common amenities in general in the useful dataframe, then I'll take a look at the most common amenities in the most frequent neighbourhoods in Seattle listings to see of there is any difference.

# In[28]:


pd.Series(' '.join(useful_df['amenities']).lower().split(',')).value_counts()[:20].plot(kind='bar')


# In[29]:


q2_df = useful_df[useful_df['neighbourhood'].isin(hoods_lst)]


# In[30]:


(useful_df.shape, q2_df.shape)


# In[31]:


pd.Series(' '.join(q2_df['amenities']).lower().split(',')).value_counts()[:20].plot(kind='bar')


# From the two bar charts illustrated above, we notice that the most common amenities are almost similar between the general approach and in the top neighbourhoods, and that as I see is from two reasons, the first one is that the top 20 neighbourhoods listings numbers represent the vast majority of the total listings, and the second is that the most common amenities are so crusial amd almost no one can live without, such as kitchen, heating and wireless internet, however we notice some change in positions when the amenities become less common.

# ## Question 3: What are the properties that affects listing prices the most? In addition prices predictions of new listings.

# In[32]:


import seaborn as sns
# calculate the correlation matrix
corr = useful_df.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# From the above heat map and the below dictionary, we figure out that few numerical variables have reasonable correlation with the prices, and as we represent this in the following pair plot, we notice the positive correlation between price and the other numerical features, except for the reviews per month variable, I think the negative correlation is due to booking less expensive listings by people and as a result cheape listings gain more number of reviews.

# In[33]:


# here I will represent the numerical variables which affect the listing prices reasonably.
high_corr = {}
for col in num_df.columns:
    correlation = (num_df['price'].corr(num_df[col]))
    if correlation > 0.2 or correlation < -0.2:
        high_corr[col] = correlation
        
high_corr


# In[34]:


col = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'reviews_per_month', 'guests_included']
sns.set(style="ticks", color_codes=True)
sns.pairplot(useful_df[col].dropna())
plt.show();


# In[35]:


useful_df.columns


# Below, I will start preparing the amenities column in dummy variable form as each amenity will represnt a separate column and generate dummy variable for other categorical columns to fit our prediction model to data frame called useful_new

# In[36]:


from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer =  CountVectorizer(tokenizer=lambda x: x.split(','))
amenities = count_vectorizer.fit_transform(useful_df['amenities'])
df_amenities = pd.DataFrame(amenities.toarray(), columns=count_vectorizer.get_feature_names())
df_amenities = df_amenities.drop('',1)


# In[37]:


df_amenities.head()


# In[38]:


useful_new = useful_df.drop('amenities', axis=1)


# In[39]:


'amenities' in useful_df.columns


# In[40]:


'amenities' in useful_new.columns


# In[41]:


useful_new = pd.concat([useful_new, df_amenities], axis=1, join='inner')


# In[42]:


useful_new.head()


# 

# In[43]:


useful_new.columns


# In[44]:


useful_new['host_identity_verified'] = useful_new['host_identity_verified'].replace('f',0,regex=True)
useful_new['host_identity_verified'] = useful_new['host_identity_verified'].replace('t',1,regex=True)


# In[45]:


def clean_data(df):
    '''
    INPUT
    df - pandas dataframe 
    
    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector
    
    '''
    df = df.dropna(subset=['price'], axis=0)
    y = df['price']
    
    #Drop price column
    df = df.drop('price', axis=1)
    
     # Fill numeric columns with the mean
    num_vars = df.select_dtypes(include=['float', 'int']).columns
    for col in num_vars:
        df[col].fillna((df[col].mean()), inplace=True)
        
    # Dummy the categorical variables
    cat_vars = df.select_dtypes(include=['object']).copy().columns
    for var in  cat_vars:
        # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)
    
    X = df
    return X, y
    
#Use the function to create X and y
X, y = clean_data(useful_new)    


# In[46]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Splitting X,y to train and test data with 0.25 test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1)


# In[47]:


rfr = RandomForestRegressor(n_estimators=500, 
                               criterion='mse', 
                               random_state=3, 
                               n_jobs=-1)
# Fitting and training the model to our data
rfr.fit(X_train, y_train)
y_test_pred = rfr.predict(X_test)
rmse_rfr= (mean_squared_error(y_test,y_test_pred))**(1/2)

print('RMSE test: %.3f' % rmse_rfr)
print('R^2 test: %.3f' % (r2_score(y_test, y_test_pred)))


# In[48]:


# Generating model coefficiants data frame
coefs_df = pd.DataFrame()
coefs_df['feature'] = X_train.columns
coefs_df['coefs'] = rfr.feature_importances_
coefs_df.sort_values('coefs', ascending=False).head(20)


# So, the best results we achieved is less than 48 dollars test RMSE error on average, and the model explains 67% of the variability in listing price, by RandomForestRegressor. The results are not the greatist bu  believe more features are needed and those features should be sharp and direct.
# 
# And from the coefficiants data frame, we notice that the physical features of the rooms themselves are affecting the pricing more than other features such as the neighbourhood or low priority amenities.

# In[ ]:





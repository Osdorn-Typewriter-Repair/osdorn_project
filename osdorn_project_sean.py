#!/usr/bin/env python
# coding: utf-8

# # How can we reduce the error in Zillow's Zestimate?

# ### Paddy Doran and Sean Oslin
# November 12, 2019

#  

# # Project planning phase

# ### Project goals
# 
# 1. Determine what factor(s) is driving the difference in Zestimate to sales price (i.e 'logerror').
# 
# 2. Build an improved model to predict the logerror.

# ### Deliverables

# 1. Verbal presentation of findings
# 2. MySQL notebook with database queries that were imported into Python for analysis
# 3. README with data definitions and analysis notes
# 4. Data analysis in a Jupyter Notebook that will allow for replication of analysis
# 5. Github repository holding the analysis and supporting materials

# ## Acquisition, prep, and exploration

# ### Data source
# All data originated with the Zillow database.

# ### Python libraries used for analysis

# In[94]:


# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Wrangling
import pandas as pd
import numpy as np

# Exploring
import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import anderson
from statsmodels.formula.api import ols
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from statsmodels.graphics.gofplots import qqplot
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Visualizing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# ### Python programming for this analysis imported from associated text documents¶

# In[2]:


import env
df = pd.read_csv("zillow.csv")


#  

# ### Data preparation

# #### Force 'head' to show all columns

# In[3]:


pd.set_option('display.max_columns', None) 


# #### Remove duplicate columns

# In[4]:


def remove_dup_col(df):
    df = df.loc[:,~df.columns.duplicated()]
    return df


# In[5]:


df = remove_dup_col(df)


# #### Calculate the number and percent of missing values for each attribute

# In[6]:


def df2(df):
    num_rows_missing = df.isna().sum()
    pct_rows_missing = num_rows_missing/len(df)*100
    df_sum = pd.DataFrame()
    df_sum['num_rows_missing'] = num_rows_missing
    df_sum['pct_rows_missing'] = pct_rows_missing
    return df_sum
# df2(df)


# #### Delete rows and columns with excessing missing values

# In[7]:


def handle_missing_values(df, prop_required_column = .9, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


def data_prep(df, cols_to_remove=[], prop_required_column=.9, prop_required_row=.75):
    df.drop(columns = cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df


# In[8]:


df = data_prep(df, cols_to_remove=[], prop_required_column=.9, prop_required_row=.75)


# #### Drop unneeded columns

# In[9]:


def drop_col(df):
        df = df.drop(columns = ['calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'propertycountylandusecode',
                            'propertylandusetypeid', 'rawcensustractandblock', 'regionidcity', 'regionidzip', 
                            'censustractandblock', 'transactiondate', 'assessmentyear',
                            'roomcnt', 'regionidcounty'])
        return df


# In[10]:


df = drop_col(df)


#  

# ### Manage missing values
# 
# For land square feet, impute the missing values by creating a linear model where landtaxvaluedollarcnt is the x-variable and the output/y-variable is the estimated land square feet.

# #### Use linear model to calculate predicted values for 'lotsizesquarefeet'.

# In[11]:


x = df['landtaxvaluedollarcnt']
y = df['lotsizesquarefeet']
ols_model = ols('lotsizesquarefeet ~ landtaxvaluedollarcnt', data=df).fit()

df['yhat'] = ols_model.predict(df[['landtaxvaluedollarcnt']])


# #### If value is NaN, replace with the predicted ('yhat') value. Otherwise, keep y value.

# In[12]:


df.lotsizesquarefeet = np.where(df.lotsizesquarefeet.isna(), df.yhat, df.lotsizesquarefeet)


# #### Impute other missing values from the median value

# In[13]:


def impute_values(df):
    sqfeet = df.calculatedfinishedsquarefeet.median()
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.fillna(sqfeet)
    
    structuretaxvalue = df.structuretaxvaluedollarcnt.median()
    df.structuretaxvaluedollarcnt = df.structuretaxvaluedollarcnt.fillna(structuretaxvalue)
    
    taxvalue = df.taxvaluedollarcnt.median()
    df.taxvaluedollarcnt = df.taxvaluedollarcnt.fillna(taxvalue)
    
    landtaxvalue = df.landtaxvaluedollarcnt.median()
    df.landtaxvaluedollarcnt = df.landtaxvaluedollarcnt.fillna(landtaxvalue)
    
    return df


# #### Use linear model to calculate predicted values for 'taxamount'

# In[14]:


x = df['taxvaluedollarcnt']
y = df['taxamount']
ols_model = ols('lotsizesquarefeet ~ taxvaluedollarcnt', data=df).fit()

df['yhat'] = ols_model.predict(df[['taxvaluedollarcnt']])


# #### If value is NaN, replace with the predicted ('yhat') value. Otherwise, keep y value.¶

# In[15]:


df.taxamount = np.where(df.taxamount.isna(), df.yhat, df.taxamount)


#   

# ### Drop rows with no 'yearbuilt' date

# In[16]:


df.fillna(value=pd.np.nan, inplace=True)


# In[17]:


df = df.dropna()


# In[18]:


def drop_col2(df): #Drop additional columns that are no longer of use. 
        df = df.drop(columns = ['taxamount', 'yhat'])
        return df


# In[19]:


df = drop_col2(df)


#  

# ### Adjust data types

# In[20]:


df[['bathroomcnt', 'calculatedfinishedsquarefeet', 'bedroomcnt','fips', 'latitude', 'longitude', 'lotsizesquarefeet', 'yearbuilt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt']] =df[['bathroomcnt', 'calculatedfinishedsquarefeet', 'bedroomcnt','fips', 'latitude', 'longitude', 'lotsizesquarefeet', 'yearbuilt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt']].astype('int64')


#  

# ### Manage outliers
# 
# #### Upper outliers
# 
# While the most common value for k is 1.5, we experimented with values as high as 5 because of the very high number of outliers. We settled on 4 to eliminate many of the outliers, but not too drastically reduce the rows of data. 

# In[21]:


def get_upper_outliers(s, k):
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k):
    outlier_cols = {col + '_outliers': get_upper_outliers(df[col], k)
                   for col in df.select_dtypes('number')}
    return df.assign(**outlier_cols)
    for col in df.select_dtypes('number'):
        df[col + '_outliers'] = get_upper_outliers(df[col], k)

    return df

add_upper_outlier_columns(df, k=4)


# In[22]:


new_df = add_upper_outlier_columns(df, k = 4.0)
outlier_cols = [col for col in new_df if col.endswith('_outliers')]
for col in outlier_cols:
    print('~~~\n' + col)
    data = new_df[col][new_df[col] > 0]
    print(data.describe())
    new_df = new_df[(new_df.logerror_outliers ==0) & (new_df.lotsizesquarefeet_outliers == 0)]
    new_df = new_df[(new_df.bathroomcnt_outliers == 0) & (new_df.calculatedfinishedsquarefeet_outliers == 0) & (new_df.calculatedfinishedsquarefeet_outliers == 0)]
    new_df = new_df[(new_df.structuretaxvaluedollarcnt_outliers == 0) & (new_df.taxvaluedollarcnt_outliers == 0) & (new_df.landtaxvaluedollarcnt_outliers == 0)]
    


# #### Lower outliers
# 
# The number of lower outliers was quite small. We decided not to exclude them from the analysis.

# In[23]:


def get_lower_outliers(s, k):
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    lower_bound = q1- k * iqr
    return s.apply(lambda x: x if x < lower_bound else 0)

def add_lower_outlier_columns(df, k):
    outlier_cols = {col + '_outliers': get_lower_outliers(df[col], k)
                     for col in df.select_dtypes('number')}
    return df.assign(**outlier_cols)
    for col in df.select_dtypes('number'):
        df[col + '_outliers'] = get_lower_outliers(df3[col], k)

    return df

add_lower_outlier_columns(df, k=4)


# #### Drop outlier columns

# In[24]:


def drop_col(new_df):
    new_df = new_df.drop(columns = [
           'parcelid_outliers', 'id_outliers', 'bathroomcnt_outliers',
           'bedroomcnt_outliers', 'buildingqualitytypeid_outliers',
           'calculatedfinishedsquarefeet_outliers', 'fips_outliers',
           'heatingorsystemtypeid_outliers', 'latitude_outliers',
           'longitude_outliers', 'lotsizesquarefeet_outliers', 'unitcnt_outliers',
           'yearbuilt_outliers', 'structuretaxvaluedollarcnt_outliers',
           'taxvaluedollarcnt_outliers', 'landtaxvaluedollarcnt_outliers',
           'logerror_outliers', 'propertyzoningdesc', 'buildingqualitytypeid', 'heatingorsystemtypeid'])
    return new_df


# In[25]:


new_df = drop_col(new_df)


#  

# ### Split data into train and test for data exploration.

# In[26]:


train, test = train_test_split(new_df, test_size=.30, random_state = 123)


#  

# ### Data encoding

# #### Encode the bathroom and bedroom counts into separate variables for each size

# In[27]:


def one_hot_encode(train, test, col_name):

    encoded_values = sorted(list(train[col_name].unique()))

    train_array = np.array(train[col_name]).reshape(len(train[col_name]),1)
    test_array = np.array(test[col_name]).reshape(len(test[col_name]),1)

    ohe = OneHotEncoder(sparse=False, categories='auto')
    train_ohe = ohe.fit_transform(train_array)
    test_ohe = ohe.transform(test_array)

    train_encoded = pd.DataFrame(data=train_ohe,
                            columns=encoded_values, index=train.index)
    train = train.join(train_encoded)

    test_encoded = pd.DataFrame(data=test_ohe,
                            columns=encoded_values, index=test.index)
    test = test.join(test_encoded)

    return train, test


# In[28]:


train, test = one_hot_encode(train, test, col_name = 'bathroomcnt')


# #### Rename encoded colums. 

# In[29]:


train.rename(columns={1:'1bath', 2:'2bath', 3:'3bath', 4: '4bath', 5:'5bath', 6:'6bath', 7: '7bath'}, inplace=True)
test.rename(columns={1:'1bath', 2:'2bath', 3:'3bath', 4: '4bath', 5:'5bath', 6:'6bath', 7: '7bath'}, inplace=True)


# #### Rename encoded colums.

# In[30]:


train, test = one_hot_encode(train, test, col_name = 'bedroomcnt')


# In[31]:


train.rename(columns={1:'1bed', 2:'2bed', 3:'3bed', 4: '4bed', 5:'5bed', 6:'6bed', 7: '7bed', 8: '8bed', 9:'9bed'}, inplace=True)
test.rename(columns={1:'1bed', 2:'2bed', 3:'3bed', 4: '4bed', 5:'5bed', 6:'6bed', 7: '7bed', 8: '8bed', 9:'9bed'}, inplace=True)


# In[32]:


train.reset_index(drop = True, inplace = True)
test.reset_index(inplace = True)


#  

# ### Data scaling
# Because of the nature of the scaling we considered both MinMax and Standard scalers. We settled on Standard as it is the most frequently used. Additionally, we chose to scale only the variables that with units of dollars and square feet.

# In[33]:


scaler = StandardScaler()

train_scaled = train.copy()
test_scaled = test.copy()

train_scaled = train_scaled[['calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt']]
test_scaled = test_scaled[['calculatedfinishedsquarefeet', 'lotsizesquarefeet','structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt']]
scaler.fit(train_scaled)
scaler.fit(test_scaled)
train_scaled = scaler.transform(train_scaled)
test_scaled = scaler.transform(test_scaled)                   


# In[34]:


train_scaled = pd.DataFrame(train_scaled)
train_scaled.columns =['calculatedfinishedsquarefeet', 'lotsizesquarefeet','structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt']
test_scaled = pd.DataFrame(test_scaled)
test_scaled.columns =['calculatedfinishedsquarefeet', 'lotsizesquarefeet','structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt']


# In[35]:


train[['calculatedfinishedsquarefeet', 'lotsizesquarefeet','structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt']]= train_scaled[['calculatedfinishedsquarefeet', 'structuretaxvaluedollarcnt', 'lotsizesquarefeet','taxvaluedollarcnt', 'landtaxvaluedollarcnt']]


# #### Create dependent variable 'logerror'

# In[36]:


# X_train = train.drop(columns ='logerror')
# y_train = train[['logerror']]
# X_test = test.drop(columns ='logerror')
# y_test = test[['logerror']]


# In[37]:


# X_train.info()


#  

# ## Cluster analysis

# ### Create first set of clusters: count of bedrooms and bathrooms

# In[38]:


ks = range(1,10)
sse = []
for k in ks:
    kmeans = KMeans(n_clusters=k, random_state = 123)
    kmeans.fit(train[['bathroomcnt', 'bedroomcnt']])

    # inertia: Sum of squared distances of samples to their closest cluster center.
    sse.append(kmeans.inertia_)

print(pd.DataFrame(dict(k=ks, sse=sse)))

plt.plot(ks, sse, 'bx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('The Elbow Method to find the optimal k')
plt.show()


# In[39]:


def target_cluster(train):
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(train[['bathroomcnt', 'bedroomcnt']])
    train['cluster'] = kmeans.predict(train[['bathroomcnt', 'bedroomcnt']])
    return train


# In[40]:


train1 = target_cluster(train)


# In[41]:


train1.groupby(train1['cluster']).mean().sort_values('logerror')


# #### Number of observations per cluster

# In[42]:


train1.cluster.value_counts()


# In[43]:


train1.groupby('cluster').mean()


#  

# #### Check of validity of the clusters and X and Y variables by mapping the clusters against themselves. This pattern appears as expected. 

# In[44]:


sns.relplot(data=train1, x='bedroomcnt', y='bathroomcnt', hue='cluster', palette = sns.color_palette("RdBu", n_colors=5))
plt.title('How does bedroom and bathroom relate to clusters?', fontsize = 18)
plt.show()


# In[45]:


sns.relplot(data=train1, x='calculatedfinishedsquarefeet', y='logerror', hue='cluster', palette = sns.color_palette("RdBu", n_colors=5))
plt.title('How does finshed square feet relate to logerror?', fontsize = 18)
plt.show()


# #### Using bedroom and bathroom counts as clusters, the graph above may indicate significant groups when looking at the finished square feet and logerror.
# 
# #### We can test the if the means are different using a one-way ANOVA test. First, we must determine if the the clusters are normally distributed. 

#  

# #### Create clusters to test for normality.

# In[46]:


cluster_0 =train1[train1.cluster == 0]
cluster_1 =train1[train1.cluster == 1]
cluster_2 =train1[train1.cluster == 2]
cluster_3 =train1[train1.cluster == 3]
#cluster_4 =train1[train1.cluster == 4] Cluster 4 will not work. Unsure of error. 


#  

# #### Plot clusters using histograms to visually check if distributions are normal.  All seem to follow a Gaussian distribution. 

# In[47]:


plt.title('Check for normality for 5 logerror clusters')

plt.subplot(231)
sns.distplot(cluster_0.logerror)
plt.title('cluster 0')

plt.subplot(232)
sns.distplot(cluster_1.logerror)
plt.title('cluster 1')

plt.subplot(233)
sns.distplot(cluster_2.logerror)
plt.title('cluster 2')

plt.subplot(234)
sns.distplot(cluster_3.logerror)
plt.title('cluster 3')

# plt.subplot(235)
# sns.distplot(cluster_4.logerror)
# plt.title('cluster 4')

plt.show()


# #### Plot clusters using a Q-Q plot to visually check if distributions are normal. All clusters have questionable Gaussian distribution.

# In[48]:


qqplot(cluster_0.logerror, line='s')
plt.show()


# In[49]:


qqplot(cluster_1.logerror, line='s')
plt.show()


# In[50]:


qqplot(cluster_2.logerror, line='s')
plt.show()


# In[51]:


qqplot(cluster_3.logerror, line='s')
plt.show()


#  

# #### Run 2 parametric tests (Anderson-Darling and Shapiro-Wilks) to check if distributions are normal.
# 
# None of the clusters appear to be normally distributed. (Only tests for cluster_0 are shown below.)
# 
# Conclusion: normality cannot be assumed using parametric tests. The one-way ANOVA test cannot be run to test the means of the clusters against each other. 

# In[52]:


result = anderson(cluster_0.logerror)
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
	else:
		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))


# In[53]:


stat, p = shapiro(cluster_0.logerror)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian')
else:
	print('Sample does not look Gaussian')


#  

# #### Using bedroom and bathroom counts as clusters, none of the X and Y pairings below demonstrated any visually significant relationships. 

# In[54]:


sns.relplot(data=train1, x='lotsizesquarefeet', y='logerror', hue='cluster', palette = sns.color_palette("RdBu", n_colors=5))
plt.title('How does lot size relate to logerror?', fontsize = 18)
plt.show()


# In[55]:


sns.relplot(data=train1, x='yearbuilt', y='logerror', hue='cluster', palette = sns.color_palette("RdBu", n_colors=5))
plt.title('How does year built relate to logerror?', fontsize = 18)
plt.show()


# In[56]:


sns.relplot(data=train1, x='structuretaxvaluedollarcnt', y='logerror', hue='cluster', palette = sns.color_palette("RdBu", n_colors=5))
plt.title('How does tax amount on the structure relate to logerror?', fontsize = 18)
plt.show()


# In[57]:


sns.relplot(data=train1, x='landtaxvaluedollarcnt', y='logerror', hue='cluster', palette = sns.color_palette("RdBu", n_colors=5))
plt.title('How does land tax amount relate to logerror?', fontsize = 18)
plt.show()


# In[58]:


sns.relplot(data=train, x='taxvaluedollarcnt', y='logerror', hue='cluster', palette = sns.color_palette("RdBu", n_colors=5))
plt.title('How does total tax amount relate to logerror?', fontsize = 18)
plt.show()


# In[59]:


sns.relplot(data=train1, x='longitude', y='latitude', hue='cluster', palette = sns.color_palette("RdBu", n_colors=5))
plt.title('How does latitude and longitude relate to clusters?', fontsize = 18)
plt.show()


#  

# ### Create second set of clusters: latitude and longitude
# 
# No apparent patterns were seen in the clusters

# In[60]:


ks = range(1,10)
sse = []
for k in ks:
    kmeans = KMeans(n_clusters=k, random_state = 123)
    kmeans.fit(train[['latitude', 'longitude']])

    # inertia: Sum of squared distances of samples to their closest cluster center.
    sse.append(kmeans.inertia_)

print(pd.DataFrame(dict(k=ks, sse=sse)))

plt.plot(ks, sse, 'bx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('The Elbow Method to find the optimal k')
plt.show()


# In[61]:


def target_cluster(train):
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(train[['latitude', 'longitude']])
    train['cluster'] = kmeans.predict(train[['latitude', 'longitude']])
    return train


# In[62]:


train2 = target_cluster(train)


# In[63]:


train2.groupby(train2['cluster']).mean().sort_values('taxvaluedollarcnt')


#  

# #### Number of observations per cluster

# In[64]:


train2.cluster.value_counts()


# In[65]:


sns.relplot(data=train2, x='bedroomcnt', y='logerror', hue='cluster', palette = sns.color_palette("RdBu", n_colors=4))
plt.title('How does the number of bedrooms relate to logerror?', fontsize = 18)
plt.show()


# In[66]:


sns.relplot(data=train2, x='lotsizesquarefeet', y='logerror', hue='cluster', palette = sns.color_palette("RdBu", n_colors=4))
plt.title('How does lot size relate to logerror?', fontsize = 18)
plt.show()


# In[67]:


sns.relplot(data=train2, x='taxvaluedollarcnt', y='lotsizesquarefeet', hue='cluster', palette = sns.color_palette("RdBu", n_colors=4))

plt.title('How does tax value relate to logerror?', fontsize = 18)
plt.show()


# In[68]:


sns.relplot(data=train2, x='yearbuilt', y='lotsizesquarefeet', hue='cluster', palette = sns.color_palette("RdBu", n_colors=4))

plt.title('How does year built relate to lot size?', fontsize = 18)
plt.show()


# In[69]:


sns.relplot(data=train2, x='yearbuilt', y='taxvaluedollarcnt', hue='cluster', palette = sns.color_palette("RdBu", n_colors=4))

plt.title('How does year built relate to tax value?', fontsize = 18)
plt.show()


#  

# ### Histograms of all variables

# In[70]:


train.hist(figsize=(20, 20), bins=10, log=False)
plt.suptitle('Histograms of all variables', fontsize = 22)
plt.show()


#  

# ### Jointplots (scatter plot and histograms) of select variables do not show any obvous relationships.

# In[71]:


sns.jointplot(data=train, x='calculatedfinishedsquarefeet', y='yearbuilt', alpha = .3)
plt.title('Square feet vs. year built', fontsize = 18)
plt.show()


# In[72]:


sns.jointplot(data=train, x='lotsizesquarefeet', y='yearbuilt', alpha = .3)
plt.title('Square feet vs. year built', fontsize = 18)
plt.show()


# In[73]:


sns.jointplot(data=train, x='calculatedfinishedsquarefeet', y='structuretaxvaluedollarcnt', alpha = .3)
plt.title('Square feet vs. building value', fontsize = 18)
plt.show()


#  

# ## Correlation matrix/heatmap of all variables

# In[74]:


corr1 = train[['bathroomcnt', 'bedroomcnt',
       'calculatedfinishedsquarefeet','latitude', 'longitude', 'lotsizesquarefeet','yearbuilt', 'structuretaxvaluedollarcnt',
       'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'logerror']]
plt.figure(figsize=(20,10))
sns.heatmap(corr1.corr(), cmap='RdYlBu', annot=True, center=0)
plt.show()


# ### T-tests

# #### T-test 1

# H<sub>0</sub> : The logerror for single-unit properties with 1 bathroom is the same as properties with 2 bathrooms
# 
# H<sub>a</sub> : The logerror for single-unit properties with 1 bathroom is NOT the same as properties with 2 bathrooms

# In[75]:


one_bathroom = train[train['bathroomcnt']==1.0]
three_bathroom = train[train['bathroomcnt']==2.0]
stats.ttest_ind(one_bathroom['logerror'], three_bathroom['logerror'])


#  

# In[76]:


plt.title('Bathroom count vs. logerror', fontsize = 18)
sns.barplot(x=train["bathroomcnt"], y=train["logerror"])
plt.show()


# #### Reject H<sub>0</sub>. The logerror for single-unit properties with 1 bathroom is NOT the same as properties with 2 bathrooms.
# 
# More generalized, the logerror is statistically different for all bathroom counts except 6.

#   

# #### T-test 2

# H<sub>0</sub> : The logerror for single-unit properties with 1 bedroom is the same as properties with 3 bedrooms
# 
# H<sub>a</sub> : The logerror for single-unit properties with 1 bedroom is NOT the same as properties with 3 bedrooms

# In[77]:


one_bedroom = train[train['bedroomcnt']==1.0]
three_bedroom = train[train['bedroomcnt']==5.0]
stats.ttest_ind(one_bedroom['logerror'], three_bedroom['logerror'])


# In[78]:


plt.title('Bedroom count vs. logerror', fontsize = 18)
sns.barplot(x=train["bedroomcnt"], y=train["logerror"])
plt.show()


# #### Fail to reject H<sub>0</sub>.  The logerror for single-unit properties with 1 bedroom is the same as properties with 2 bedrooms.
# 
# More generalized, the logerror is statistically different for for all single-unit residences with 5 or more bedrooms.

# In[79]:


X_train = train.drop(columns ='logerror')
y_train = train[['logerror']]
X_test = test.drop(columns ='logerror')
y_test = test[['logerror']]


# In[80]:


# train.drop(columns = (['bathroomcnt', 'bedroomcnt']))
# test.drop(columns = (['bathroomcnt', 'bedroomcnt']))


# In[81]:


predictions=pd.DataFrame({'actual':y_train['logerror']}).reset_index(drop=True)
predictions['baseline'] = y_train.mean()[0]
predictions.head()


# In[82]:


lm1=LinearRegression()
lm1.fit(X_train[['bathroomcnt', 'bedroomcnt']],y_train)
lm1_predictions=lm1.predict(X_train[['bathroomcnt', 'bedroomcnt']])
predictions['lm1']=lm1_predictions


# In[83]:


MSE_baseline = mean_squared_error(predictions.actual, predictions.baseline)
SSE_baseline = MSE_baseline*len(predictions.actual)
RMSE_baseline = sqrt(MSE_baseline)
r2_baseline = r2_score(predictions.actual, predictions.baseline)
print("MSE =",MSE_baseline, "SSE =", SSE_baseline, "RMSE =", RMSE_baseline, "R2 =", r2_baseline)


# In[84]:


MSE_1 = mean_squared_error(predictions.actual, predictions.lm1)
SSE_1 = MSE_1*len(predictions.actual)
RMSE_1 = sqrt(MSE_1)
r2_1 = r2_score(predictions.actual, predictions.lm1)
print("MSE =", MSE_1, "SSE=", SSE_1, "RMSE=", RMSE_1, "R2=", r2_1)


# In[85]:


model=lm1.predict(X_test[['bathroomcnt', 'bedroomcnt']])
model=model.ravel().reshape(11550)
y_test1=np.array(y_test).ravel().reshape(11550)
best_model=pd.DataFrame({'predictions':model,'logerror':y_test1})

best_model.head()


# In[86]:


pd.DataFrame({'actual': predictions.actual,
              'lm1': predictions.lm1,
              'lm_baseline': predictions.baseline.ravel()})\
.melt(id_vars=['actual'], var_name='model', value_name='prediction')\
.pipe((sns.relplot, 'data'), x='actual', y='prediction', hue='model')
min = -.5
max = .5
plt.plot([min, max],[min, max], c='red', ls=':')
plt.ylim(min, max)
plt.xlim(min, max)
plt.title('Predicted vs Actual Log Error')


# ## Look at that difference!

# In[90]:


from sklearn.feature_selection import SelectKBest, f_regression, RFE
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, LinearRegression

def optimal_features(X_train, X_test, y_train, number_of_features):
    '''Taking the output of optimal_number_of_features, as n, and use that value to 
    run recursive feature elimination to find the n best features'''
    cols = list(X_train.columns)
    model = LinearRegression()
    
    #Initializing RFE model
    rfe = RFE(model, number_of_features)

    #Transforming data using RFE
    train_rfe = rfe.fit_transform(X_train,y_train)
    test_rfe = rfe.transform(X_test)
    
    #Fitting the data to model
    model.fit(train_rfe, y_train)
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index
    
    X_train_rfe = pd.DataFrame(train_rfe, columns=selected_features_rfe)
    X_test_rfe = pd.DataFrame(test_rfe, columns=selected_features_rfe)
    
    return selected_features_rfe, X_train_rfe, X_test_rfe


# In[91]:


optimal_features(X_train, X_test, y_train, 14)
X_train_rfe = optimal_features(X_train, X_test, y_train, 12)[1]
X_test_rfe = optimal_features(X_train, X_test, y_train, 12)[2]
# X_train_rfe


# In[92]:


lm2 =LinearRegression()
lm2.fit(X_train_rfe,y_train)
lm2_predictions=lm2.predict(X_train_rfe)
predictions['lm2']=lm2_predictions

lm2_y_intercept = lm2.intercept_
# print("intercept: ", lm1_y_intercept)
lm2_coefficients = lm2.coef_
# print("coefficients: ", lm1_coefficients)


# In[93]:


MSE_2 = mean_squared_error(predictions.actual, predictions.lm2)
SSE_2 = MSE_2*len(predictions.actual)
RMSE_2 = sqrt(MSE_2)
r2_2 = r2_score(predictions.actual, predictions.lm2)
print("MSE =", MSE_2, "SSE=", SSE_2, "RMSE=", RMSE_2, "R2=", r2_2)


# __Model is sliiiiightly improved by the optimal features function. Feature engineering is required to seek further improvement.__

# In[106]:


rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',min_samples_leaf=3,n_estimators=100,max_depth=2, random_state=123)


# In[107]:


dtx = X_train[['latitude', 'longitude', 'bathroomcnt', 'bedroomcnt']]
dty = train.cluster


# In[108]:


rf.fit(dtx,dty)


# In[109]:


print(rf.feature_importances_)
y_pred = pd.DataFrame(rf.predict(dtx))
y_pred_proba = rf.predict_proba(dtx)
rf.score(dtx,dty)
print(classification_report(dty,y_pred))


# In[ ]:


000


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Importing telcom data set 
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
db=pd.read_csv('E:\\DATA science\\Intership Programm\\telcom_data.csv', na_values=['?',None])
db.head()


# In[3]:


# getting the number of data points in the data
print(f" There are {db.shape[0]} rows and {db.shape[1]} columns ")


# In[4]:


# column Names

db.columns.tolist()


# ## Analysing Dataset

# In[5]:


# identifying the top 10 handsets used by the customers.

top_10_handsets = db['Handset Type'].value_counts().head(10)

print("Top 10 handsets used by customers:")
print(top_10_handsets)


# In[6]:


# Visualizing Tope 10 handsets used by customers
top_10_handsets.plot(kind='bar', figsize=(6, 6))
plt.title('Top 10 Handsets Used by Customers')
plt.xlabel('Handset Type')
plt.ylabel('Count')
plt.show()


# In[7]:


# Calculate the percentage of each handset
handset_percentages = (top_10_handsets / top_10_handsets.sum()) * 100

# Create a pie chart to visualize the top 10 handsets
plt.pie(handset_percentages, labels=top_10_handsets.index, autopct='%1.1f%%')
plt.title('Top 10 Handsets Used by Customers (Percentage)')
plt.show()


# ## Identify the top 3 handset manufacturer

# In[8]:


# identifying the top 3 handset manufacturers 

top_3_handsets_manufacturers = db['Handset Manufacturer'].value_counts().head(3)

print("Top 3 handset manufacturers:")
print(top_3_handsets_manufacturers)


# In[9]:


# Visualizing top 3 handset manufacturers
top_3_handsets_manufacturers.plot(kind='bar', figsize=(6, 6),color='red')
plt.title('Top 3 handset manufacturers:')
plt.xlabel('Manufacturer')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# In[10]:


# Calculate the percentage of each manufacturer
manufacturer_percentages = (top_3_handsets_manufacturers / top_3_handsets_manufacturers.sum()) * 100

# Create a pie chart to visualize the top 10 manufacturers
plt.pie(manufacturer_percentages, labels=top_3_handsets_manufacturers.index, autopct='%1.1f%%', radius=1, colors=['y', 'g', 'b'])
plt.title('Top 3 Handset Manufacturers (Percentage)')
plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.


# ## Identify the top 5 handsets per top 3 handset manufacturer

# In[11]:


# identifying the top 5 handsets per top 3 handset manufacturer

top_3_manufacturers = db['Handset Manufacturer'].value_counts().head(3).index.tolist()

top_5_per_manufacturer = {}

for manufacturer in top_3_manufacturers:
    subset = db[db['Handset Manufacturer'] == manufacturer]
    top_5_handsets = subset['Handset Type'].value_counts().head(5)
    top_5_per_manufacturer[manufacturer] = top_5_handsets

# Printing the top 5 handsets for each top 3 manufacturer
for manufacturer, top_5 in top_5_per_manufacturer.items():
    print(f"Top 5 handsets for {manufacturer}:")
    for handset, count in top_5.items():
        print(f"Handset: {handset}, Count: {count}")
    print()


# In[12]:


# Coverting dictionary in to dataframe for ploting bar graph

top_5_per_Manufacturer = pd.DataFrame(top_5_per_manufacturer)


# In[13]:


top_5_per_Manufacturer.plot(kind='bar', stacked=False)
plt.xlabel('Manufacturer')
plt.ylabel('Count')
plt.title('Top 5 Handsets per Top 3 Handset Manufacturer')
plt.xticks(rotation=90)
plt.legend(title='Handset Type')
plt.show()


# ## Task 1.1 

# In[14]:


# Grouping the data by 'MSISDN/Number' (assuming 'MSISDN/Number' is the user identifier)
user_info = db.groupby('MSISDN/Number').agg({
    'Bearer Id': 'count',  # Number of xDR sessions
    'Dur. (ms)': 'sum',    # Session duration
    'Total DL (Bytes)': 'sum',  # Total download data
    'Total UL (Bytes)': 'sum'   # Total upload data
})


# In[15]:


# Adding total data volume for each application
app_columns = [col for col in db.columns if 'Total DL' in col or 'Total UL' in col]
app_data = db.groupby('MSISDN/Number')[app_columns].sum()

# Concatenating the aggregated data
user_info = pd.concat([user_info, app_data], axis=1)

# Renaming the columns for application data
user_info = user_info.rename(columns={
    'Total DL (Bytes)': 'Total_DL',
    'Total UL (Bytes)': 'Total_UL'
})

# Displaying the aggregated information
print(user_info)


# ## Interpretation and Recommendation for marketing teams
# 
# 1. As we can see Top three Manufrectures are Apple, Samsung,Huawei.
# 2. After analysing top 10 Handset used by customers bar graph we can clearly see that most of our customers are Huawei user and the most used brand in huawei is Handset: Huawei B528S-23A followed by Apple iPhone 6S (A1688) but the count difference is approxmatly 10k.
# 3. By leveraging this analysis, marketing teams can better align their efforts to consumer preferences, enhance customer engagement, and potentially increase market share by focusing on popular handsets and aligning marketing efforts to customer preferences and trends.    
#     
# 
# 

# ## Task 1.2 
# 
# ### Handling Missing values and Data cleaning

# In[16]:


# Getting dimensions of the dataset

db.shape


# In[17]:


## Displaying the first few rows to understand the structure and types of data.

db.head(5)


# In[18]:


##Identifying columns, their data types (numeric, categorical), and their names.

db.info()


# ## Description of Dataset

# In[19]:


db.describe()


# In[20]:


## Identifying missing values in the dataset

db.isnull().sum()


# In[21]:


plt.figure(figsize=(16,9))
sns.heatmap(db.isnull())
plt.title('Missing Values Heatmap')
plt.show()


# In[22]:


# Getting the total count of rows in the DataFrame
total_count = len(db)

# Calculate the percentage of missing values for each column
missing_percentage = ((db.isnull().sum()) / total_count) * 100

# Print the percentage of missing values for each column
print(missing_percentage)


# In[23]:


missing=pd.DataFrame((db.isnull().sum()*100)/db.shape[0]).reset_index()
plt.figure(figsize=(20,5))
ax=sns.pointplot('index',0,data=missing)
plt.xticks(rotation=90,fontsize=7)
plt.title("percentage of missing values")
plt.ylabel("percentage")
plt.show()


# In[24]:


# Before filling 'NA' values will created copy of dataset to keep our maine dataset safe

db1=db.copy()


# ## Partitioning the dataset into separate entities based on numerical and categorical values for targeted analysis and distinct treatment.

# In[25]:


# Select numerical columns
numerical_db = db1.select_dtypes(include='float')

# Select categorical/object columns
categorical_db = db1.select_dtypes(exclude='float')


# ## Handling Missing values of numerical columns 

# In[26]:


# Filling numerical missing values with Mean
missing_v=numerical_db.isnull().sum()
for col in missing_v.index:
    numerical_db[col].fillna(numerical_db[col].mean(), inplace=True)


# In[27]:


missing=pd.DataFrame((numerical_db.isnull().sum()*100)/numerical_db.shape[0]).reset_index()
plt.figure(figsize=(20,5))
ax=sns.pointplot('index',0,data=missing)
plt.xticks(rotation=90,fontsize=7)
plt.title("percentage of missing values")
plt.ylabel("percentage")
plt.show()


# ## Handling Missing values of Object Columns

# In[28]:


missing_values_per_column_object = (categorical_db.isnull().sum() / categorical_db.shape[0]) * 100


# In[29]:


missing=pd.DataFrame((categorical_db.isnull().sum()*100)/categorical_db.shape[0]).reset_index()
plt.figure(figsize=(20,5))
ax=sns.pointplot('index',0,data=missing)
plt.xticks(rotation=60,fontsize=17)
plt.title("percentage of missing values")
plt.ylabel("percentage")
plt.show()


# In[30]:


missing_val=categorical_db.isna().sum()
for col in missing_val.index:
    categorical_db[col].fillna(categorical_db[col].mode()[0], inplace=True)


# In[31]:


missing=pd.DataFrame((categorical_db.isnull().sum()*100)/categorical_db.shape[0]).reset_index()
plt.figure(figsize=(20,5))
ax=sns.pointplot('index',0,data=missing)
plt.xticks(rotation=60,fontsize=17)
plt.title("percentage of missing values")
plt.ylabel("percentage")
plt.show()


# ## Updating missing values in db1 with the values from numerical_db and numerical_db

# In[32]:


db1.update(numerical_db)
db1.update(categorical_db)


# In[33]:


db1.isnull().sum()


# In[34]:


# comparing missing value of earlier dataset and after removing missing values by using heatmap

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) #subplot 1
sns.heatmap(db.isnull())
plt.title('Heatmap before Handling Missing Values')

plt.subplot(1,2,2) #subplot 2
sns.heatmap(db1.isnull())
plt.title('Heatmap after Handling Missing Values')

plt.tight_layout()
plt.show()


# ## Bivariate Analysis

# In[35]:


## exploring the relationship between each application & the total DL+UL data using appropriate methods and interpret your findings.


# In[36]:


db1['Total_DL_UL'] = db1['Total UL (Bytes)'] + db1['Total DL (Bytes)']
correlation_matrix = db1.corr()
correlation_matrix


# In[37]:


correlation_with_total_data = correlation_matrix['Total_DL_UL']


# In[38]:


for app in db1.columns[:-3]:  # Exclude columns for total DL, total UL, and the newly added total DL+UL
    plt.figure(figsize=(6, 4))
    plt.scatter(db1[app], db1['Total_DL_UL'], alpha=0.5)
    plt.title(f"Relationship between {app} and Total DL+UL Data")
    plt.xlabel(app)
    plt.ylabel("Total DL+UL Data")
    plt.grid(True)
    plt.show()


# ## Variable transformations – segment the users into the top five decile classes based on the total duration for all sessions and compute the total data (DL+UL) per decile class. 

# ### Compute Total Session Duration for Each User:

# In[39]:


# Group by user and calculate total session duration
total_duration_per_user = db1.groupby('MSISDN/Number')['Dur. (ms).1'].sum().reset_index()


# ### Segment Users into Decile Classes:

# In[40]:


# Calculate deciles based on session duration
total_duration_per_user['DecileClass'] = pd.qcut(total_duration_per_user['Dur. (ms).1'], q=10, labels=False)


# In[41]:


# Merge data_usage with total_duration_per_user based on MSISDN/Number
merged_data = pd.merge(total_duration_per_user, db1, on='MSISDN/Number')

# Calculate total data (DL+UL) per decile class
total_data_per_decile = merged_data.groupby('DecileClass')['Total_DL_UL'].sum()


# In[42]:


db1.head(5)


# ## 	Correlation Analysis – computing a correlation matrix for the following variables and interpret your findings: Social Media data, Google data, Email data, Youtube data, Netflix data, Gaming data, and Other data 

# In[43]:


selected_columns=['Social Media DL (Bytes)', 'Social Media UL (Bytes)','Youtube DL (Bytes)','Youtube UL (Bytes)','Netflix DL (Bytes)','Netflix UL (Bytes)','Google DL (Bytes)','Google UL (Bytes)','Email DL (Bytes)'
                , 'Email UL (Bytes)','Gaming DL (Bytes)','Gaming UL (Bytes)','Other UL (Bytes)','Other DL (Bytes)']


# In[44]:


selected_data = db1[selected_columns]


# In[45]:


# Calculate the correlation matrix
correlation_matrix = selected_data.corr()


# In[46]:


# Display the correlation matrix
print(correlation_matrix)


# ### Standardizing the dataset

# In[47]:


from sklearn.preprocessing import StandardScaler

# Selecting only numeric columns for standardization
numeric_columns = db1.select_dtypes(include=['float64', 'int64']).columns

# Standardize only the numeric columns
scaler = StandardScaler()
standardized_data = scaler.fit_transform(db1[numeric_columns])

# 'standardized_data' contains the standardized version of numeric columns in 'db1'


# In[48]:


from sklearn.decomposition import PCA

pca = PCA()
pca.fit(standardized_data)

# Explained Variance Ratio

explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance.cumsum()

# Visualize Explained Variance

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()


# ## User Engagement Analysis

# ### Session Frequency

# In[49]:


session_frequency = db1.groupby(by=['MSISDN/Number'])['Dur. (ms)'].transform('count')


# In[50]:


len(session_frequency)


# ### calculating duration of the session

# In[51]:


Session_Duration = db1['Dur. (ms)']


# ### Calculating session total traffic (download and upload (bytes))

# In[52]:


total_traffic = db1['Total DL (Bytes)'] + db1['Total UL (Bytes)']


# ### Visualizing Total Traffic vs Session Duration

# In[53]:


plt.figure(figsize=(10, 6))
plt.scatter(Session_Duration, total_traffic)
plt.title('Total Traffic vs. Session Duration')
plt.xlabel('Session Duration')
plt.ylabel('Total Traffic')
plt.show()


# ### Session Count per Customer (MSISDN):

# In[54]:


# Coverting 'MSISDN/Number' column in string type
db1['MSISDN/Number'] = db1['MSISDN/Number'].astype(str)

session_count = db1.groupby('MSISDN/Number')['Bearer Id'].nunique().reset_index(name='SessionCount')
top_10_session_count = session_count.nlargest(10, 'SessionCount')

# Plotting the top 10 session counts for customers
plt.figure(figsize=(10, 6))
plt.bar(top_10_session_count['MSISDN/Number'], top_10_session_count['SessionCount'], color='skyblue')
plt.xlabel('MSISDN/Number')
plt.ylabel('Session Count')
plt.title('Top 10 Customers by Session Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Total Duration per Customer (MSISDN)

# In[55]:


total_duration = db1.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='TotalDuration')
top_10_total_duration = total_duration.nlargest(10, 'TotalDuration')

# Plotting the top 10 customers by total duration
plt.figure(figsize=(10, 6))
plt.bar(top_10_total_duration['MSISDN/Number'], top_10_total_duration['TotalDuration'], color='lightgreen')
plt.xlabel('MSISDN/Number')
plt.ylabel('Total Duration (ms)')
plt.title('Top 10 Customers by Total Duration')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Total Traffic (Download + Upload) per Customer (MSISDN)

# In[56]:


db1['TotalTraffic'] = db1['Total DL (Bytes)'] + db1['Total UL (Bytes)']
total_traffic = db1.groupby('MSISDN/Number')['TotalTraffic'].sum().reset_index(name='TotalTraffic')
top_10_total_traffic = total_traffic.nlargest(10, 'TotalTraffic')
print(top_10_total_traffic)


# In[57]:


# Visualizing Total Traffic

plt.figure(figsize=(10, 6))
plt.bar(top_10_total_traffic['MSISDN/Number'], top_10_total_traffic['TotalTraffic'], color='salmon')
plt.xlabel('MSISDN/Number')
plt.ylabel('Total Traffic (Bytes)')
plt.title('Top 10 Customers by Total Traffic')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Normalizing each engagement metric and run a k-means (k=3) to classify customers into three groups of engagement. 

# In[58]:


from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

# Selecting only numeric columns for standardization
numeric_columns = db1.select_dtypes(include=['float64', 'int64']).columns

# Standardize only the numeric columns
scaler = StandardScaler()
standardized_data = scaler.fit_transform(db1[numeric_columns])

# 'standardized_data' contains the standardized version of numeric columns in 'db1'


# In[59]:


from sklearn.preprocessing import MinMaxScaler

# Selecting only numeric columns
numeric_columns = db1.select_dtypes(include=['float64', 'int64'])

# Check for missing values and handle them if present
numeric_columns.fillna(numeric_columns.mean(), inplace=True)  # Replace missing values with mean

# Normalize the engagement metrics
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(numeric_columns)
normalized_df = pd.DataFrame(normalized_data, columns=numeric_columns.columns)


# ### K-means Clustering

# In[60]:


from sklearn.cluster import KMeans

# Perform k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=44)
clusters = kmeans.fit_predict(normalized_df)

# Add the clusters as a new column in the original DataFrame
db1['Cluster'] = clusters
print(db1['Cluster'].value_counts())


# Grouping DataFrame by 'Cluster' column and finding top 10 customers for each cluster
top_10_customers_per_cluster = db1.groupby('Cluster')['MSISDN/Number'].value_counts().groupby(level=0).nlargest(10)

# Printing top 10 customers in each cluster
for cluster, top_10_customers in top_10_customers_per_cluster.groupby(level=0):
    print(f"\nTop 10 customers in Cluster {cluster}:")
    print(top_10_customers)


# ### Compute Non-normalized Metrics for Each Cluster

# In[61]:


# Group by cluster and compute statistics for each cluster
cluster_stats = db1.groupby('Cluster').agg(['min', 'max', 'mean', 'sum'])

# Loop through each column and get statistics for each metric
for column in db1.columns:
    if column in cluster_stats.columns.levels[0]:
        for stat in ['min', 'max', 'mean', 'sum']:
            print(f"Metric: {column}, Stat: {stat}")
            print(cluster_stats[column][stat])
            print("\n")


# ### Aggregating Total Traffic per Application & Finding Top 10 Engaged Users per Application

# In[62]:


# Select columns related to applications traffic
app_traffic_columns = [
    'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
    'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
    'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
    'Google DL (Bytes)', 'Google UL (Bytes)',
    'Email DL (Bytes)', 'Email UL (Bytes)',
    'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
    'Other DL (Bytes)', 'Other UL (Bytes)'
]

# Sum the traffic for each application
db1['Total_App_Traffic'] = db1[app_traffic_columns].sum(axis=1)

# Group by application and derive top 10 most engaged users per application
top_10_users_per_app = {}
for app_column in app_traffic_columns:
    app_name = app_column.split(' ')[0]  # Extracting application name from column title
    top_users = db1.groupby('MSISDN/Number')[app_column].sum().nlargest(10)
    top_10_users_per_app[app_name] = top_users

# Displaying top 10 most engaged users per application
for app_name, top_users in top_10_users_per_app.items():
    print(f"Top 10 users for {app_name}:")
    print(top_users)
    print("\n")


# In[63]:


for app_name, top_users in top_10_users_per_app.items():
    plt.figure(figsize=(8, 6))
    plt.pie(top_users.values, labels=top_users.index, autopct='%1.1f%%')
    plt.title(f"Top 10 Users for {app_name}")
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


# ### Ploting the top 3 most used applications  using Bar plot

# In[64]:


total_traffic_per_app = db1[app_traffic_columns].sum()

# Get the top 3 most used applications
top_3_apps = total_traffic_per_app.nlargest(3)

# Plotting the top 3 most used applications
plt.figure(figsize=(8, 6))
top_3_apps.plot(kind='bar', color='skyblue')
plt.title('Top 3 Most Used Applications')
plt.xlabel('Application')
plt.ylabel('Total Traffic')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Ploting pie chart alos to visualized Top 3 Used application

# In[65]:


plt.figure(figsize=(8, 8))
plt.pie(top_3_apps, labels=top_3_apps.index, autopct='%1.1f%%', startangle=140)
plt.title('Top 3 Most Used Applications')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.show()


# In[66]:


from sklearn.cluster import KMeans

# Define a range of values for k
k_values = range(1, 11)
wcss = []  # Within-cluster sum of squares

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_df)
    wcss.append(kmeans.inertia_)  # inertia_ gives the WCSS for the model

# Plotting the Elbow Method curve
plt.figure(figsize=(8, 6))
plt.plot(k_values, wcss, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[67]:


wcss


# In[68]:


db1.columns


# In[72]:


db1_engagement_cluster = db1[['MSISDN/Number', 'Cluster']]
db1_engagement_cluster.to_csv('Cluster.csv', index=False)


# In[ ]:





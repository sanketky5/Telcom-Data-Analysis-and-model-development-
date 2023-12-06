#!/usr/bin/env python
# coding: utf-8

# ## Task 3 - Experience Analytics

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


# ### Partitioning the dataset into separate entities based on numerical and categorical values for targeted analysis and distinct treatment.

# In[3]:


# Select numerical columns
numerical_db = db.select_dtypes(include='float')

# Select categorical/object columns
categorical_db = db.select_dtypes(exclude='float')


# In[4]:


# Filling numerical missing values with Mean value
missing_v=numerical_db.isnull().sum()
for col in missing_v.index:
    numerical_db[col].fillna(numerical_db[col].mean(), inplace=True)


# In[5]:


# Filling Categorical missing values with mode value
missing_val=categorical_db.isna().sum()
for col in missing_val.index:
    categorical_db[col].fillna(categorical_db[col].mode()[0], inplace=True)


# ### Updating missing values in db1 with the values from numerical_db and numerical_db

# In[6]:


db.update(numerical_db)
db.update(categorical_db)


# In[7]:


db.isnull().sum()


# In[8]:


def replace_outliers_with_mean(column):
    if pd.api.types.is_numeric_dtype(column):
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Check if there are outliers in the column
        outliers = column[(column < lower_bound) | (column > upper_bound)]
        
        # If outliers exist, replace them with the mean value
        if not outliers.empty:
            column = column.where(~((column < lower_bound) | (column > upper_bound)), column.mean())
        
    return column


# In[9]:


db1 = db.apply(replace_outliers_with_mean)

# Print the modified DataFrame with outliers replaced by mean values
print(db)


# In[10]:


# Grouping by 'MSISDN/Number' and aggregating the required columns
grouped_data = db.groupby('MSISDN/Number').agg({
    'TCP DL Retrans. Vol (Bytes)': 'mean',
    'TCP UL Retrans. Vol (Bytes)': 'mean',
    'Avg RTT DL (ms)': 'mean',
    'Avg RTT UL (ms)': 'mean',
    'Avg Bearer TP DL (kbps)': 'mean',
    'Avg Bearer TP UL (kbps)': 'mean',
    'Handset Type': lambda x: x.mode().iloc[0] if not x.mode().empty else None
}).reset_index()

# Calculate Average TCP Retransmission considering both DL and UL for each user
grouped_data['Average TCP Retransmission'] = (grouped_data['TCP DL Retrans. Vol (Bytes)'] + grouped_data['TCP UL Retrans. Vol (Bytes)']) / 2

# Calculate Average RTT for each user
grouped_data['Average RTT'] = (grouped_data['Avg RTT DL (ms)'] + grouped_data['Avg RTT UL (ms)']) / 2

# Calculate Average Throughput considering both DL and UL for each user
grouped_data['Average Throughput'] = (grouped_data['Avg Bearer TP DL (kbps)'] + grouped_data['Avg Bearer TP UL (kbps)']) / 2

# Display the calculated values per user
print(grouped_data[['MSISDN/Number', 'Average TCP Retransmission', 'Average RTT', 'Handset Type', 'Average Throughput']])


# ### Visualizing Handset Types

# In[11]:


# Count the occurrences of each handset type
top_n = 15  # Change this value to display a different number of top handset types
top_handsets = grouped_data['Handset Type'].value_counts().nlargest(top_n)

# Visualizing Top N Handset Types
plt.figure(figsize=(10, 8))
sns.barplot(x=top_handsets.values, y=top_handsets.index, palette='viridis')
plt.xlabel('Count')
plt.ylabel('Handset Type')
plt.title(f'Top {top_n} Most Common Handset Types')
plt.show()


# ### Task 3.2 - Computing & listing 10 of the top, bottom and most frequent:
# a.	TCP values in the dataset. 
# b.	RTT values in the dataset.
# c.	Throughput values in the dataset.
# 

# In[12]:


# Function to compute top N values
def top_n_values(data, column, n=10):
    return data[column].nlargest(n)

# Function to compute bottom N values
def bottom_n_values(data, column, n=10):
    return data[column].nsmallest(n)

# Function to compute most frequent values
def most_frequent_values(data, column, n=10):
    return data[column].value_counts().head(n)

# Compute top, bottom, and most frequent TCP values
top_10_tcp = top_n_values(grouped_data, 'Average TCP Retransmission')
bottom_10_tcp = bottom_n_values(grouped_data, 'Average TCP Retransmission')
most_frequent_tcp = most_frequent_values(grouped_data, 'Average TCP Retransmission')

# Compute top, bottom, and most frequent RTT values
top_10_rtt = top_n_values(grouped_data, 'Average RTT')
bottom_10_rtt = bottom_n_values(grouped_data, 'Average RTT')
most_frequent_rtt = most_frequent_values(grouped_data, 'Average RTT')

# Compute top, bottom, and most frequent Throughput values
top_10_throughput = top_n_values(grouped_data, 'Average Throughput')
bottom_10_throughput = bottom_n_values(grouped_data, 'Average Throughput')
most_frequent_throughput = most_frequent_values(grouped_data, 'Average Throughput')

# Display the computed values
print("Top 10 TCP Values:\n", top_10_tcp)
print("Bottom 10 TCP Values:\n", bottom_10_tcp)
print("Most Frequent TCP Values:\n", most_frequent_tcp)

print("\nTop 10 RTT Values:\n", top_10_rtt)
print("Bottom 10 RTT Values:\n", bottom_10_rtt)
print("Most Frequent RTT Values:\n", most_frequent_rtt)

print("\nTop 10 Throughput Values:\n", top_10_throughput)
print("Bottom 10 Throughput Values:\n", bottom_10_throughput)
print("Most Frequent Throughput Values:\n", most_frequent_throughput)


# In[13]:


grouped_data.columns


# In[14]:


# Function to plot top, bottom, and most frequent values
def plot_values(values, title):
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(values)), values, color='skyblue')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.show()

# Plotting TCP values
plot_values(top_10_tcp, 'Top 10 TCP Values')
plot_values(bottom_10_tcp, 'Bottom 10 TCP Values')
plot_values(most_frequent_tcp, 'Most Frequent TCP Values')

# Plotting RTT values
plot_values(top_10_rtt, 'Top 10 RTT Values')
plot_values(bottom_10_rtt, 'Bottom 10 RTT Values')
plot_values(most_frequent_rtt, 'Most Frequent RTT Values')

# Plotting Throughput values
plot_values(top_10_throughput, 'Top 10 Throughput Values')
plot_values(bottom_10_throughput, 'Bottom 10 Throughput Values')
plot_values(most_frequent_throughput, 'Most Frequent Throughput Values')


# In[15]:


import matplotlib.pyplot as plt

# Function to plot box plots for top values
def plot_boxplot(data, title):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data)
    plt.title(title)
    plt.xticks([1, 2, 3], ['TCP', 'RTT', 'Throughput'])
    plt.ylabel('Values')
    plt.grid(True)
    plt.show()

# Extracting top values for TCP, RTT, and Throughput
top_values = [top_10_tcp.values, top_10_rtt.values, top_10_throughput.values]

# Plotting box plots for top values
plot_boxplot(top_values, 'Top Values: TCP, RTT, and Throughput')


# ### Task 3.3 - Computing & reporting following
# 
# 1.The distribution of the average throughput per handset type and provide interpretation for your findings.
# 
# 2.The average TCP retransmission view per handset type and provide interpretation for your findings.

# In[16]:


grouped_data.columns


# ### Visualize the distribution of average throughput per handset type:

# In[17]:


# Calculate the top 10 most frequent handset types
top_handset_types = grouped_data['Handset Type'].value_counts().head(10).index.tolist()

# Filter the dataframe to include only the top handset types
filtered_data = grouped_data[grouped_data['Handset Type'].isin(top_handset_types)]

plt.figure(figsize=(12, 6))
sns.barplot(data=filtered_data, x='Handset Type', y='Average Throughput', palette='viridis')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.title('Average Throughput per Top Handset Types')
plt.xlabel('Handset Type')
plt.ylabel('Average Throughput')
plt.tight_layout()
plt.show()


# ### Findings
# 
# As we already know the most users are using Huawei B2585-23A Handset and there are receving more no of Average Throughput from our network as compare to other handset types .
# 

# In[18]:


#2.The average TCP retransmission view per handset type and provide interpretation for your findings.


# In[19]:


# Calculate the top 10 most frequent handset types
top_handset_types = grouped_data['Handset Type'].value_counts().head(10).index.tolist()

# Filter the dataframe to include only the top handset types
filtered_data = grouped_data[grouped_data['Handset Type'].isin(top_handset_types)]

plt.figure(figsize=(12, 6))
sns.barplot(data=filtered_data, x='Handset Type', y='Average TCP Retransmission', palette='viridis')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.title('Average Throughput per Top Handset Types')
plt.xlabel('Handset Type')
plt.ylabel('Average Throughput')
plt.tight_layout()
plt.show()


# ### Findings 
# 
# According to Above graph we can clearly see that users with Huawei are facing more TCP Retransmission other devies such as apple and Samsung handset have less no of TCP Retransmission as compared to Huawei B52285-23A so we need to work on toward that direction

# ### Task 3.4 - Using the experience metrics above, perform a k-means clustering (where k = 3) to segment users into groups of experiences and provide a brief description of each cluster. (The description must define each group based on your understanding of the data)

# In[26]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select relevant features for clustering
features = ['Average TCP Retransmission', 'Average RTT', 'Average Throughput']

# Subsetting data with selected features
cluster_data = grouped_data[features].copy()

# Scale the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Applying KMeans with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Assigning cluster labels to the data
cluster_data['experience_Cluster'] = cluster_labels

# Analyzing and describing clusters
cluster_summary = cluster_data.groupby('experience_Cluster').mean()

# Displaying the cluster summaries
print(cluster_summary)


# ### Cluster Analysis:
# 
# #####  Cluster 0:
# 
# Average TCP Retransmission: 8.43e+06
# 
# Average RTT: 70.71 ms
# 
# Average Throughput: 2159.20 kbps
# 
# * Interpretation:
# 
# Performance: Moderate TCP retransmission with relatively lower RTT and low throughput.
# 
# Network Experience: Users in this cluster experience moderate issues with TCP retransmission, implying occasional data packet 
# retransmissions. The average throughput is low to moderate, and the latency (RTT) is relatively better compared to other clusters.
# 
# ##### Cluster 1:
# 
# Average TCP Retransmission: 1.01e+07
# 
# Average RTT: 51.32 ms
# 
# Average Throughput: 27526.99 kbps
# 
# * Interpretation:
# 
# Performance: High TCP retransmission with low RTT and significantly higher throughput.
# 
# Network Experience: Users in this cluster exhibit high TCP retransmission rates, indicating frequent packet retransmissions. However, they experience lower latency (RTT) and significantly higher throughput, suggesting a relatively better network capacity to handle data transmission.
# 
# ###### Cluster 2:
# 
# Average TCP Retransmission: 1.10e+09
# 
# Average RTT: 71.93 ms
# 
# Average Throughput: 35860.75 kbps
# 
# * Interpretation:
# 
# Performance: Extremely high TCP retransmission with moderate RTT and very high throughput.
# 
# Network Experience: Users in this cluster face severe issues with TCP retransmission, indicating a large number of packet retransmissions. Despite moderate latency (RTT), users experience very high throughput, suggesting that while the network capacity is good, there might be significant issues with data reliability due to high retransmissions.
# 

# #### Insights:
# 
# Cluster 0 represents users with moderate issues in network performance.
# 
# Cluster 1 represents users with good network performance despite high TCP retransmissions.
# 
# Cluster 2 represents users facing severe issues with TCP retransmissions, despite high throughput.

# In[37]:


experience_Cluster = cluster_data['experience_Cluster']
experience_Cluster.to_csv('experience_Cluster.csv', index=False)


# In[33]:


original_data=db.to_csv('original_data.csv', index=False)


# In[38]:


cluster_data['experience_Cluster'].isnull().sum()


# In[ ]:





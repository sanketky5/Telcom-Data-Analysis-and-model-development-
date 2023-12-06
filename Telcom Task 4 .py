#!/usr/bin/env python
# coding: utf-8

# ## Task 4 - Satisfaction Analysis

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


original_data = pd.read_csv('C:\\Users\\Sanket\\AI course DIGIChrome\\Intership Programm\\original_data.csv')


# In[3]:


experience_Cluster = pd.read_csv('C:\\Users\\Sanket\\AI course DIGIChrome\\Intership Programm\\experience_Cluster.csv')


# In[4]:


engagement_cluster = pd.read_csv('C:\\Users\\Sanket\\AI course DIGIChrome\\Intership Programm\\Cluster.csv')


# In[5]:


print(original_data.index)
print(experience_Cluster.index)
print(engagement_cluster.index)


# In[6]:


common_index = range(min(len(original_data), len(experience_Cluster), len(engagement_cluster)))

# Reindexing DataFrames to have the same index range
original_data = original_data.reindex(common_index)
experience_Cluster = experience_Cluster.reindex(common_index)
engagement_cluster = engagement_cluster.reindex(common_index)


# In[7]:


original_data['experience_column'] = experience_Cluster['experience_Cluster']
original_data['engagement_column'] = engagement_cluster['Cluster']


# In[8]:


engagement_cluster.columns


# In[9]:


experience_Cluster.columns


# In[10]:


engagement_columns = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
                      'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 'Social Media DL (Bytes)',
                      'Social Media UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)',
                      'Netflix UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
                      'Email UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)',
                      'Other UL (Bytes)']

experience_columns = ['HTTP DL (Bytes)', 'HTTP UL (Bytes)',
                      'Activity Duration DL (ms)', 'Activity Duration UL (ms)']


# In[11]:


from sklearn.cluster import KMeans
from scipy.spatial import distance

# Calculate centroids of the clusters
engagement_centroid = original_data.groupby('engagement_column')[engagement_columns].mean()  
experience_centroid = original_data.groupby('experience_column')[experience_columns].mean()

# Calculate engagement score
original_data['engagement_score'] = original_data.apply(lambda row: distance.euclidean(row[engagement_columns], engagement_centroid.loc[row['engagement_column']]), axis=1)

# Calculate experience score
original_data['experience_score'] = original_data.apply(lambda row: distance.euclidean(row[experience_columns], experience_centroid.loc[row['experience_column']]), axis=1)


# ### Calculating Satisfaction Score:

# In[12]:



original_data['satisfaction_score'] = (original_data['engagement_score'] + original_data['experience_score']) / 2


# In[13]:


original_data['satisfaction_score']


# ### Geting top 10 satisfied customers

# In[14]:


top_10_satisfied = original_data.nlargest(10, 'satisfaction_score')


# In[15]:


top_10_satisfied


# In[16]:


plt.figure(figsize=(10, 6))
plt.barh(range(len(top_10_satisfied)), top_10_satisfied['satisfaction_score'], color='pink')
plt.yticks(range(len(top_10_satisfied)), top_10_satisfied['MSISDN/Number'])  # Custom y-axis ticks and labels
plt.xlabel('Satisfaction Score')
plt.ylabel('MSISDN/Number')
plt.title('Top 10 Satisfied Customers')
plt.gca().invert_yaxis()  # Invert y-axis to display highest score at the top
plt.show()


# ### Building regression model for predicting the satisfaction score of a customer

# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


features = ['engagement_score', 'experience_score']  # Actual column names

# Splitting the data into features (X) and target variable (y)
X = original_data[features]
y = original_data['satisfaction_score']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[18]:


y_test


# In[19]:


y_pred


# In[21]:


#Visualizing model test data against result data 
plt.scatter(y_test,y_pred)
plt.show()


# ### model deployment and model testing by entering engagement score and experience score get the value of statisfaction score

# In[23]:


import pickle

pickle.dump(model,open('model_telco.pkl','wb'))


# In[24]:


model_telco=pickle.load(open('model_telco.pkl','rb'))


# In[25]:


print(model.predict([[4.539153e+08,7.242196e+10]]))


# ### Runing a k-means (k=2) on the engagement & the experience score

# In[26]:


from sklearn.cluster import KMeans

# Assuming 'engagement_score' and 'experience_score' are columns in original_data
engagement_experience = original_data[['engagement_score', 'experience_score']]

# Run KMeans with k=2
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(engagement_experience)

# Get the cluster labels for each data point
cluster_labels = kmeans.labels_

# Add cluster labels to the DataFrame
original_data['cluster_labels'] = cluster_labels


# ### Aggregating the average satisfaction & experience score per cluster. 

# In[27]:


# 'cluster_labels' contains the cluster labels obtained from k-means clustering

# Group by 'cluster_labels' and calculate average satisfaction & experience scores
cluster_scores = original_data.groupby('cluster_labels')[['satisfaction_score', 'experience_score']].mean()

print(cluster_scores)


# In[28]:


# Plotting the average scores per cluster
cluster_scores.plot(kind='bar', figsize=(8, 6))
plt.title('Average Satisfaction and Experience Scores per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Score')
plt.xticks(rotation=0)
plt.legend(['Satisfaction Score', 'Experience Score'])
plt.show()


# In[29]:


original_data


# In[30]:


table_db=original_data[['MSISDN/Number','engagement_score','experience_score','satisfaction_score']]


# In[31]:


table_db


# In[32]:


table_db=pd.DataFrame(table_db)
table_db


# ### Exporting Table to Sql

# In[33]:


pip install pyodbc


# In[34]:


import pyodbc


# In[35]:


conn=pyodbc.connect('Driver={SQL Server};'
                   'Server=DESKTOP-49OUI7N\SQLEXPRESS;'
                   'Database=newdb;'
                   'Trusted_Connection=yes;')


# In[36]:


# Create a cursor object
cursor = conn.cursor()

# Define the SQL command to create a new table based on DataFrame columns and data types
create_table_query = '''
    CREATE TABLE Final_Telcom_table (
        MSISDN_Number NUMERIC,
        engagement_score NUMERIC,
        experience_score NUMERIC,
        satisfaction_score NUMERIC
    )
'''

# Execute the SQL command to create the new table
cursor.execute(create_table_query)

# Commit the changes
conn.commit()

# Insert data from the DataFrame into the newly created table
for index, row in table_db.iterrows():
    cursor.execute('INSERT INTO Final_Telcom_Table VALUES (?, ?, ?, ?)', tuple(row))

# Commit the changes after insertion
conn.commit()

# Close the connection
conn.close()


# ### ScreenShot of of a select query output on the exported table

# ![image.png](attachment:image.png)

# In[ ]:





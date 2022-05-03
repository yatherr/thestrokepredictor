# Import our libraries

# Import pandas and numpy 
import pandas as pd
import numpy as np

import pickle

import matplotlib.pyplot as plt

# Helper function to split our data 
from sklearn.model_selection import train_test_split 

# Import our Logistic Regression model 
from sklearn.linear_model import LogisticRegression

# Import helper functions to evaluate our model 
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score

# Import z-score helper function
import scipy.stats as stats

from IPython.display import Image

# Import helper functipn for hyper-parameter tuning 
from sklearn.model_selection import GridSearchCV

# Import Decision Tree
# from sklearn.tree import DecisionTreeClassifier

# Import Random Forest 
from sklearn.ensemble import RandomForestClassifier

# Import metrics to score our model 
from sklearn import metrics

# LOAD IN AND CLEAN UP THE DATA BEFORE MERGING
# Load in the first stroke dataset
df = pd.read_csv('https://raw.githubusercontent.com/yatherr/Graduation-/main/healthcare-dataset-stroke-data.csv')

# Drop the id column
df.drop(columns=['id'], inplace=True)

# Fill the bmi null values in df
df['bmi'] = df.bmi.fillna(df.bmi.mean())

# Remove entries with gender Other from df 
df = df[df['gender'] != 'Other']

# Normalize our numerical features to ensure they have equal weight when I build my classifiers
# Create a new column for normalized age
df['age_norm']=(df['age']-df['age'].min())/(df['age'].max()-df['age'].min())

# Create a new column for normalized avg glucose level
df['avg_glucose_level_norm']=(df['avg_glucose_level']-df['avg_glucose_level'].min())/(df['avg_glucose_level'].max()-df['avg_glucose_level'].min())

# Create a new column for normalized bmi
df['bmi_norm']=(df['bmi']-df['bmi'].min())/(df['bmi'].max()-df['bmi'].min())

# Load in the second stroke dataset
df2 = pd.read_csv('https://raw.githubusercontent.com/yatherr/Graduation-/main/train_strokes.csv')

# Drop the id column
df2.drop(columns=['id'], inplace=True)

# Fill the bmi null values in df2
df2['bmi'] = df2.bmi.fillna(df2.bmi.mean())

# Create a new category for the smoking null values
df2['smoking_status'] = df2['smoking_status'].fillna('not known')

# Remove entries with gender Other from df2
df2 = df2[df2['gender'] != 'Other']

# Normalize our numerical features to ensure they have equal weight when I build my classifiers
# Create a new column for normalized age
df2['age_norm']=(df2['age']-df2['age'].min())/(df2['age'].max()-df2['age'].min())

# Create a new column for normalized avg glucose level
df2['avg_glucose_level_norm']=(df2['avg_glucose_level']-df2['avg_glucose_level'].min())/(df2['avg_glucose_level'].max()-df2['avg_glucose_level'].min())

# Create a new column for normalized bmi
df2['bmi_norm']=(df2['bmi']-df2['bmi'].min())/(df2['bmi'].max()-df2['bmi'].min())

# Merge the two df's
df_master = df.merge(df2, how='outer') 


# EXTRACT ALL STROKE ENTRIES AND ISOLATE 1000 RANDOM NON-STROKE ENTRIES INTO A DF
# Create a df from dataset with just the stroke entries 
s_df = df_master.loc[df_master['stroke'] == 1]

# Remove age outliers from s_df
s_df = s_df.loc[s_df['age'] >= 45]

# Create a df from the dataset with the no stroke entries 
n_df = df_master.sample(n=1100, random_state=30)
n_df = n_df.loc[n_df['stroke'] == 0] 

# Merge them
df_final = s_df.merge(n_df, how='outer')

# FEATURE ENGINEERING TIME
# Convert certain features into numerical values
df_final = pd.get_dummies(df_final, columns=['gender', 'Residence_type', 'smoking_status', 'ever_married', 'work_type'])

# Begin to train our model
selected_features = ['age', 'bmi', 'avg_glucose_level', 'hypertension', 'heart_disease']

X = df_final[selected_features]

y = df_final['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

 # RANDOM FOREST CLASSIFIER
 # Init our Random Forest Classifier Model 
#model = RandomForestClassifier()

params = {
    'n_estimators' : [10, 50, 100],
    'criterion' : ['gini', 'entropy'],
    'max_depth': [5, 10, 100, None], 
    'min_samples_split': [2, 10, 100],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search_cv = GridSearchCV( 
    estimator=RandomForestClassifier(), 
    param_grid=params,
    scoring='accuracy' )

# fit all combination of trees. 
grid_search_cv.fit(X_train, y_train)

#  the highest accuracy-score. 
model = grid_search_cv.best_estimator_

# Fit our model 
model.fit(X_train, y_train)

# Save our model using pickle
pickle.dump(model, open('models/rfc.pkl', 'wb') )

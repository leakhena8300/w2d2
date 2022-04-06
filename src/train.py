#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#get_ipython().system('sudo apt-get install build-essential swig')
#get_ipython().system('curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install')
#get_ipython().system('pip install auto-sklearn')
#get_ipython().system('pip install pipelineprofiler # visualize the pipelines created by auto-sklearn')
#get_ipython().system('pip install shap')
#get_ipython().system('pip install --upgrade plotly')
#get_ipython().system('pip3 install -U scikit-learn')


# In[ ]:


import datetime
import logging
import logging.config
from joblib import dump
import shap


# **Option and Setting**

# In[ ]:


model_path = "/content/drive/MyDrive/Introduction2DataScience/tutorials/models/"


# In[ ]:


timesstr = str(datetime.datetime.now()).replace(' ', '_')


# In[ ]:


log_config = {
    "version":1,
    "root":{
        "handlers" : ["console"],
        "level": "DEBUG"
    },
    "handlers":{
        "console":{
            "formatter": "std_out",
            "class": "logging.StreamHandler",
            "level": "DEBUG"
        }
    },
    "formatters":{
        "std_out": {
            "format": "%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(lineno)d : (Process Details : (%(process)d, %(processName)s), Thread Details : (%(thread)d, %(threadName)s))\nLog : %(message)s",
            "datefmt":"%d-%m-%Y %I:%M:%S"
        }
    },
}


# In[ ]:


logging.config.dictConfig(log_config)


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

#from google.colab import drive
#drive.mount('/content/drive/')
#get_ipython().magic('cd /content/drive/My Drive/Introduction2DataScience/data/')

df = pd.read_csv(
    "data/banking.csv"
)
df.insert(0, 'customer_id', range(1,41189))
df.columns
df.head(10)


# In[ ]:


df_numbers = df.copy()
from collections import Counter
for col in df.columns:
  cnt = Counter(df[col].tolist())
  dicty = {}
  for numb, k in enumerate(cnt.keys()):
    dicty[k] = numb
  df_numbers[col] = df[col].apply(lambda x: dicty[x] if isinstance(x, str) else x)


# In[ ]:


#preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
target = df_numbers["y"]
del df_numbers["y"]
x_train,x_test,y_train,y_test = train_test_split(df_numbers, target, test_size= 0.2)
x_train.shape, x_test.shape,y_train.shape,y_test.shape


# In[ ]:


test_size = 0.2
logging.info(f'train test split with test_size={test_size}')


# In[ ]:


#creating of model
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
clf = tree.DecisionTreeClassifier()
scaler = StandardScaler()
transf = PolynomialFeatures(degree = 1)
model = Pipeline(steps=[ 
                        ('scaler', scaler),
                        ('tranform', transf),
                        ('regressor', clf)])


# In[ ]:


dump(model, f'{model_path}model{timesstr}.pkl')


# In[ ]:


#training and evaluating
cross_val_score(model,x_train,y_train)


# In[ ]:


model.fit(x_train, y_train)


# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, plot_confusion_matrix
plot_confusion_matrix(model, x_test, y_test)


# In[ ]:


explainer = shap.KernelExplainer(model = model.predict, data = x_test.iloc[:50, :], link = "identity")


# In[42]:


# Set the index of the specific example to explain
X_idx = 0
shap_value_single = explainer.shap_values(X = x_test.iloc[X_idx:X_idx+1,:], nsamples = 100)
x_test.iloc[X_idx:X_idx+1,:]
# print the JS visualization code to the notebook
shap.initjs()
shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_value_single,
                features = x_test.iloc[X_idx:X_idx+1,:], 
                show=False,
                matplotlib=True
                )
plt.savefig(f"{model_path}shap_example_{timesstr}.png")
logging.info(f"Shapley example saved as {model_path}shap_example_{timesstr}.png")


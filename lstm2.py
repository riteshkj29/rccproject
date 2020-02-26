import inline as inline
import numpy as np  # For mathematical calculations
import keras.models
from keras.layers import Dense
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt
import seaborn as sn  # For plotting graphs
import re
import nltk
# %matplotlib inline
import warnings  # To ignore any warnings

warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pandas import DataFrame

le = LabelEncoder()

x1 = pd.ExcelFile("/Users/ritesh_kj/Desktop/Projpython/agr_en_train.xls")
print(x1.sheet_names)
df = x1.parse("agr_en_train")
df.columns = ['id', 'text', 'label', 'extra']
df.drop('extra', axis=1)
df.head()
df['text'] = df['text'].apply(
    lambda x: re.sub(r'\b[A-Z]+\b', '', re.sub('\s+', ' ', re.sub(r'[^a-zA-Z]', ' ', str(x)))))
df['text'] = df['text'].apply(lambda x: x.lower())
identity_columns = ['OAG', 'NAG', 'CAG']
for key in identity_columns:
    df.label[df.label == key] = identity_columns.index(key) + 1
print(len(identity_columns))

df2 = pd.DataFrame({'text': df['text'].replace(r'\n', ' ', regex=True),
                    'label': LabelEncoder().fit_transform(df['label'].replace(r' ', '', regex=True)),
                    })

# Creating train and val dataframes
X_train, X_test, y_train, y_test = train_test_split(df2["text"].values, df2["label"].values, test_size=0.2,
                                                    random_state=42)
X_train, y_train = df2["text"].values, df2["label"].values

testpd = pd.read_csv("/Users/ritesh_kj/Desktop/Projpython/agr_en_dev.csv", encoding='latin-1')
dataset=testpd.values
testpd.columns = ['id', 'text', 'label']
testpd['text'] = testpd['text'].apply(
    lambda x: re.sub(r'\b[A-Z]+\b', '', re.sub('\s+', ' ', re.sub(r'[^a-zA-Z]', ' ', x))))
testpd['text'] = testpd['text'].apply(lambda x: x.lower())
test = pd.DataFrame({'text': testpd['text'].replace(r'\n', ' ', regex=True)})
test = test["text"].values
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
model = keras.models.Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
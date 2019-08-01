# Importing the Dataset
import pandas as pd
data = pd.read_excel('review.xlsx')
import matplotlib.pyplot as plt
import numpy as np
X = data.iloc[:, [0]].values
X=X.astype('int')
y = data.iloc[:, [-1]].values

from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
y=labelencoder_x.fit_transform(y) 
y.reshape(-1,1)
y.astype('int')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)


from sklearn.neighbors import KNeighborsRegressor 
classifier = KNeighborsRegressor(n_neighbors=3)  
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)
print("PREDICTIONS",y_pred)
def mov(rate):
    y_pred = classifier.predict([[rate]])
    print("PREDICTIONS",y_pred)
    y_pred=y_pred.astype('int')
    for pred in y_pred:
        if pred==1:
            print("BLOCK_BURSTER-PREDICTION--","*","SUPER-FLOP")
        elif pred==2:
            print("BLOCK_BURSTER-PREDICTION--","**","FLOP")
        elif pred==3:
            print("BLOCK_BURSTER-PREDICTION--","***","AVERAGE")
        elif pred==4:
            print("BLOCK_BURSTER-PREDICTION--","****",'HIT')
        elif pred==5:
            print("BLOCK_BURSTER-PREDICTION--","****",'SUPER-HIT')
            
print("\n1.SUPER-FLOP \n2.FLOP \n3.AVERAGE \n4.HIT \n5.SUPER-HIT")
mov(int(input("Enter your Rating for movie----->")))
plt.plot(y_pred)

import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv("passengers.csv")
#print(passengers)

# Update sex column to numerical
passengers["Sex"] = passengers['Sex'].map({'male':'0','female':'1'})

# Fill the nan values in the age column
mean_values = passengers["Age"].mean()
passengers["Age"].fillna(value = mean_values,inplace = True)
#print(passengers)
# Create a first class column
passengers['Firstclass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)

# Create a second class column
passengers['Secondclass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)

#print(passengers)
# Select the desired features
features = passengers[['Sex','Age', 'Firstclass', 'Secondclass','Fare']]
survived = passengers["Survived"]

# Perform train, test, split
f_train,f_test,s_train,s_test = train_test_split(features,survived,train_size = 0.8)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
f_train = scaler.fit_transform(f_train)
f_test = scaler.transform(f_test)

# Create and train the model
model = LogisticRegression()
model.fit(f_train,s_train)

# Score the model on the train data
#print(model.score(f_train,s_train))

# Score the model on the test data
#print(model.score(f_test,s_test))

# Analyze the coefficients
#print(model.coef_)
#print(list(zip(['Sex','Age','FirstClass','SecondClass','Fare'],model.coef_[0])))

# Sample passenger features
#Jack = np.array([0.0,20.0,0.0,0.0,50.0])
#Rose = np.array([1.0,17.0,1.0,0.0,100.0])
#You = np.array([1.0,15.0,0.0,1.0,125.0])

# Combine passenger arrays
sample_passengers = np.array([ Jack , Rose, You ])

# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)

# Make survival predictions!
#print(model.predict(sample_passengers))
#print(model.predict_proba(sample_passengers))


print("Welcome to the Titanic Survival Prediction !!")
print("This model will tel you ehether you woukd have survives on the Titanic by analysing some the inputs you provide")
#Name = ("What was your name :- ")
Gender = input("Enter your gender (male/female) :- ")
Age = input("Enter your age :- ")
First_class= int(input("Did you belong to 1st class (1/0) :- "))
Second_class= int(input("Did you belong to 2nd class (1/0) :- "))
Fare = input("What was your voyage fare :- ")

name = np.array([Gender,Age,First_class,Second_class,Fare])
passenger = sclaer.transform(name)

prediction = model.predict(passenger)
if prediction == 1:
  print("Yes, you would have survived during the voyage !!")
else:
  print("Sorry, you would have died on the voyage !!")
  


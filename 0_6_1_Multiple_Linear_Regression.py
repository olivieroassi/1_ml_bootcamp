############################DEV VERSION#####################

######################################MASTERCLEAN / PRODUCTION########################################################

#####PREPROCESSING   
#import the dataset
#pddf(rdspend,damin,mark,state,profit)(50,5)
import pandas as pd 
pddf  = pd.read_csv(r"C:\Users\Oliver\Downloads\forever\020618_bootcamp_ml\6_0_Multiple_Linear_Regression_50_Startups.csv")

#split into dependent and independent matrix/array
#x(rdspend,admin,mark,state)(50,4)
x = pddf.iloc[:,:-1].values 
#y(profit)(50,1)
y = pddf.iloc[:,4].values

#encode categorical values into binary dummy variables
#x(california,florida,newyork,rdspend,admin,mark)(50,6)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:,3] = labelencoder_X.fit_transform(x[:,3])   
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#avoid dummy variables trap
#x(florida,ny,rdspend,admin,mark)(50,5)
x = x[:, 1:] #select c1toTheEnd, get rid of california

#split into training and test set
#x_train(40,5), y_train(40,1)
#x_test(10,5), y_test(10,1)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0 )

#####PROCESSING
#Create a multiple linear regressor and fit it to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(x_train,y_train) 

####MODEL ASSESSMENT
#compare y to y_pred
#y_pred(50,1)
y_pred = regressor.predict(x)
pddf.assign(y_train = pddf.Profit, y_pred = y_pred, diff = y_pred - pddf.Profit )

#compare y_train to y_train_pred
#y_train_pred(40,1)
y_train_pred = regressor.predict(x_train)
y_train


#predict the test set results and compare it to y_pred
#predict : y_test_pred(10,1)
y_test_pred = regressor.predict(x_test)
#compare to : y_test(10,1)
y_test_pred
y_test

#####REPORTING PORTFOLIO
    

    
           
        
    
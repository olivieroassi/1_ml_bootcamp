#EDA with excel
    #startup(rdspend,administration,marketingspend,state,profit)(50 startups,5 attributes)
    #
#EDA
    #import libraries
    import pandas as pd 
    #import and inspect dataset pddf(rdspend f, admin f, mark f, state categ, profit f)(50,5)
    import pandas as pd 
    pddf  = pd.read_csv(r"C:\Users\Oliver\Downloads\forever\020618_bootcamp_ml\6_0_Multiple_Linear_Regression_50_Startups.csv")
    pddf.head(5)
    pddf.tail(5)
    pddf.columns
    pddf.dtypes
    pddf.shape
    #already cleaned for eda for manager
    #eda for data scientist
        #transfo
    describe = pddf.describe()
        #visu : HG,BC,BP,Scatterplot
    
#Multiple Linear Regression
        
#####################################PREPROCESSING#############################################################
    #preprocessing for MLR
        #missing values
            #no missing values
        #feature selection (boruta)
        #create dependent and independent matrix/array
            #x(rdspend,admin,mark,state)(50,4)
                x = pddf.iloc[:,:-1].values 
                    #object type means , many type in the matrix, can not be seen in the variable explorer
            #y(profit)(50,1)
                y = pddf.iloc[:,4].values
        #encode categorical values into binary dummy variables
            #x(california,florida,newyork,rdspend,admin,mark)(50,6)
                from sklearn.preprocessing import LabelEncoder, OneHotEncoder
                labelencoder_X = LabelEncoder()
                x[:,3] = labelencoder_X.fit_transform(x[:,3])   
                onehotencoder = OneHotEncoder(categorical_features = [3]) #encode column 3, State as dummy variable
                x = onehotencoder.fit_transform(x).toarray()
        #avoid dummy variables trap
            #x(florida,ny,rdspend,admin,mark)(50,5)
                x = x[:, 1:] #select c1toTheEnd, get rid of california
            #Actually the library avoids automatically the dummy trap, but it is good to be aware of!
        #split into training and test set
            #x_train(40,5), y_train(40,1)
            #x_test(10,5), y_test(10,1)
                from sklearn.cross_validation import train_test_split
                x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0 )
        #feature scaling
            #take into account by the library
############################################PROCESSING#######################################################
    #processing for MLR
        #Build a model : MLR with a backward elimination feature selection
            from sklearn.linear_model import LinearRegression
            regressor = LinearRegression() #create object of LinearRegressionClass
            regressor.fit(x_train,y_train) #fit the Multiple linear regressor on the training set
    #model assessment, test the performance of our model, predict the test set results
        #compare y to y_pred
            #y_pred(50,1)
                y_pred = regressor.predict(x)
                pddf.assign(y_train = pddf.Profit, y_pred = y_pred, diff = y_pred - pddf.Profit )
        #compare y_train to y_train_pred
            #y_train_pred(40,1)
                y_train_pred = regressor.predict(x_train)
                y_train
            #visualization : More than 2 variables, dimension : impossible
        #predict the test set results and compare it to y_pred
            #predict : y_test_pred(10,1)
                y_test_pred = regressor.predict(x_test)
            #compare to : y_test(10,1)
                y_test_pred
                y_test
                
    #reporting
    

    
           
        
    
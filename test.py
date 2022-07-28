from PreProcessingFile  import *


X_test,data=preProcessing("AmazonProductClassification.csv")
scaler = StandardScaler()
scaler.fit(X_test)
X_test = scaler.transform(X_test)

knn_from_pickle = pickle.load(open('saved_model_logistic.sav', 'rb'))
accuracy=np.mean(knn_from_pickle.predict(X_test) ==  data["ProductGrade"])*100
print ("The achieved accuracy is " + str(accuracy))
from PreProcessingFile import *


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores
X1,data=preProcessing("AmazonProductClassification.csv")
print("Features selected After correlation: ",X1.columns)
# plot for all features
plt.subplots(figsize=(12, 8))
sns.heatmap(data[["price", "manufacturer", "number_of_reviews", "number_of_answered_questions",
                  "number_available_in_stock", "sellercount", "bestsellerRank", "categoryRank", "avgprice",
                  "stock_type", "LastSubCat", "ProductGrade"]].corr(), annot=True)
plt.show()

Y = data["ProductGrade"]

scaler = StandardScaler()
scaler.fit(X1)
X1 = scaler.transform(X1)
X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.20, shuffle=True, random_state=1)

C =  100  # SVM regularization parameter'
svc_ovr = OneVsRestClassifier(svm.SVC(kernel='linear', C=C)).fit(X_train, y_train)
svc_ovo = OneVsOneClassifier(svm.SVC(kernel='linear', C=C)).fit(X_train, y_train)

bf_svc_ovr = OneVsRestClassifier(svm.SVC(kernel='rbf', gamma=0.1, C=C)).fit(X_train, y_train)
starttime = time.time()
bf_svc_ovo = OneVsOneClassifier(svm.SVC(kernel='rbf', gamma=0.1, C=C)).fit(X_train, y_train)
endtime = time.time()
timee = endtime - starttime

#poly_svc_ovo= svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
#saved_model_poly =  pickle.dump(poly_svc_ovo,open('saved_model_poly.sav', 'wb'))

saved_model_svc_ovr = pickle.dump(svc_ovr,open('saved_model_svc_ovr.sav', 'wb'))
saved_model_svc_ovo = pickle.dump(svc_ovo,open('saved_model_svc_ovo.sav', 'wb'))
saved_model_bf_ovr =  pickle.dump(bf_svc_ovr,open('bf_svc_ovr.sav', 'wb'))
saved_model_bf_ovo =  pickle.dump(bf_svc_ovo ,open('bf_svc_ovo.sav', 'wb'))

for i, clf in enumerate((svc_ovr,svc_ovo,bf_svc_ovr,bf_svc_ovo #,poly_svc_ovo#
                                )):
    starttime2 = time.time()
    predictions = clf.predict(X_test)
    endtime2 = time.time()
    accuracy = np.mean(predictions == y_test)
    if (i == 0):
        print("SVC with linear kernel one vs rest: ")
    elif (i == 1):
        print("SVC with linear kernel one vs one: ")
    elif (i == 2):
        print("SVC with RBF kernel one vs rest: ")
    elif (i == 3):
        print("SVC with RBF kernel one vs one: ")
        timee2 = endtime2 - starttime2
    elif (i == 4):
        print("SVC with polynomial (degree 3) kernel one vs one:")
    print(accuracy)



best_score=0

for gamma in [1, 0.1, 0.01, 0.001, 0.0001]:
    for C in [  0.1, 1,10, 100]:
        # for each combination of parameters, train an SVC
        bff_svc_ovo = OneVsOneClassifier(svm.SVC(kernel='rbf', gamma=gamma, C=C)).fit(X_train, y_train)
        predictions = bff_svc_ovo.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print('Testing Accuracy: (%.8f)   Using kernel: %s C: %s gamma: %s ' % (accuracy,"rbf", C, gamma))
        if accuracy > best_score:
            best_score = accuracy
            best_parameters = {'C': C, 'gamma': gamma}

print("Best score: {:.2f}".format(best_score))
print("Best parameters: {}".format(best_parameters))

scores2=evaluate_model(bf_svc_ovo,X_train,y_train)
print("cross validation score for RBF kernel one vs one is "+ str(abs(scores2.mean())))
print("Training time of RBF kernel(highest accuracy) :",timee)
print("Testing time of RBF kernel(highest accuracy) :",timee2)

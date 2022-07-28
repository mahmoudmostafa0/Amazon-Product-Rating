from PreProcessingFile  import *

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
X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.20, shuffle=True,random_state=1)



bdt = AdaBoostClassifier(DecisionTreeClassifier( max_depth= 3),
                         algorithm="SAMME",
                         n_estimators=100)

clf = tree.DecisionTreeClassifier(max_depth=3)
starttime = time.time()
clf.fit(X_train,y_train)
endtime = time.time()
timee = endtime - starttime
saved_model_tree = pickle.dump(clf,open('saved_model_tree.sav', 'wb'))
starttime2 = time.time()
y_prediction = clf.predict(X_test)
endtime2 = time.time()
timee2 = endtime2 - starttime2
accuracy=np.mean(y_prediction == y_test)*100
print ("The achieved accuracy using Decision Tree is " + str(accuracy))

bdt.fit(X_train,y_train)
saved_model2_adaboost = pickle.dump(bdt,open('saved_model_adaboost.sav', 'wb'))
y_prediction = bdt.predict(X_test)
accuracy=np.mean(y_prediction == y_test)*100
print ("The achieved accuracy using Adaboost is " + str(accuracy))



dt = DecisionTreeClassifier()
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]

}

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=dt, param_grid=params, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean*100, stdev, param))

print("Best Accuracy: %f using %s" % (grid_result.best_score_*100, grid_result.best_params_))

#best_score=0;
#for criterion in ["gini", "entropy"]:
    #   for max_depth in [2, 3, 5, 10, 20]:
    #   for min_samples_leaf in [5, 10, 20, 50, 100]:
    #       clf = tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,criterion=criterion)
    #       clf.fit(X_train, y_train)
    #       y_prediction = clf.predict(X_test)
    #       accuracy = np.mean(y_prediction == y_test) * 100
    #       print('Testing Accuracy: (%.8f)   Using criterion: %s max_depth: %s min_samples_leaf: %s ' % (accuracy,criterion, max_depth, min_samples_leaf))
    #       if accuracy > best_score:
    #           best_score = accuracy
#           best_parameters = {'criterion': criterion, 'max_depth': max_depth,'min_samples_leaf':min_samples_leaf}

#print("Best score: {:.2f}".format(best_score))
#print("Best parameters: {}".format(best_parameters))

print("Training time of DecisionTreeClassifier :",timee)
print("Testing time of DecisionTreeClassifier :",timee2)
scores2=evaluate_model(clf,X_train,y_train)
print("cross validation score for RBF kernel one vs one is "+ str(abs(scores2.mean())))
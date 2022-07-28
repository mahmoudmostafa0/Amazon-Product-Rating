from PreProcessingFile  import *


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

X1,data=preProcessing("AmazonProductClassification.csv")
print("Features selected After correlation: ",X1.columns)
#plot for all features
plt.subplots(figsize=(12, 8))
sns.heatmap(data[["price","manufacturer","number_of_reviews","number_of_answered_questions","number_available_in_stock","sellercount","bestsellerRank","categoryRank","avgprice","stock_type","LastSubCat","ProductGrade"]].corr(), annot=True)
plt.show()

Y = data["ProductGrade"]


scaler = StandardScaler()
scaler.fit(X1)
X1 = scaler.transform(X1)



X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.20, shuffle=True,random_state=1)
params = {
  'multi_class': ['multinomial', 'ovr'],
   'solver' :  ['newton-cg', 'lbfgs','sag'],
    'C': [100,10, 1.0, 0.1, 0.01]

}
clss = LogisticRegression( penalty='l2',max_iter=10000)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=clss, param_grid=params, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean*100, stdev, param))
# summarize results
print("Best Trainng Accuracy: %f using %s" % (grid_result.best_score_*100, grid_result.best_params_))

grid_predictions = grid_result.predict(X_test)
print('Best Testing Accuracy: %f '%metrics.accuracy_score(y_test, grid_predictions))

for c in [100,  1.0,  0.1, 0.01]:
    for penalty in  ['l1', 'l2']:
        clss = LogisticRegression(multi_class='ovr', solver='liblinear', penalty=penalty, C=c, max_iter=10000)
        clss.fit(X_train, y_train)
        prediction = clss.predict(X_test)
        score = metrics.accuracy_score(y_test, prediction)
        print(
            'Testing Accuracy: (%.8f)  Using C: %s Solver: %s Penalty: %s' % (
            score*100, c, 'liblinear',penalty))




clss = LogisticRegression(multi_class='multinomial', solver='newton-cg', penalty='l2', C=100)
start_time = time.time()
clss.fit(X_train, y_train)
end_time = time.time()
time_lapsed = end_time - start_time
saved_model_logistic = pickle.dump(clss,open('saved_model_logistic.sav', 'wb'))
y_train_predicted = clss.predict(X_train)
print('Training Accuracy with multi_class= multinomial, solver=newton-cg, penalty=l2, C=100: ', metrics.accuracy_score(y_train, y_train_predicted))
print("Training Time in seconds: ",time_lapsed)
start_time2 = time.time()
prediction = clss.predict(X_test)
end_time2 = time.time()
time_lapsed2 = end_time2 - start_time2
print('Testing Accuracy with multi_class= multinomial, solver=newton-cg, penalty=l2, C=100:: ', metrics.accuracy_score(y_test, prediction))
print("Testing Time in seconds: ",time_lapsed2)
scores2=evaluate_model(clss,X_train,y_train)
print("cross validation score is "+ str(abs(scores2.mean())))






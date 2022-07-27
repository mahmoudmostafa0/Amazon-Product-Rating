import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from Pre_processing import *
import seaborn as sns
import json
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
def mapseller(v):
    if pd.isnull(v):
        return pd.Series([np.nan, np.nan])
    try:
        vv = v.replace('=>', ':')
        dict_data = json.loads(vv)
        Sum = 0.0
        count = 1
        if isinstance(dict_data['seller'], list):
            for i in dict_data['seller']:
                hold = i['Seller_price_' + str(count)]
                Sum += float(hold)
                count += 1
        else:
            Sum = float(dict_data['seller']['Seller_price_1'])
            count += 1
        return pd.Series([Sum / (count - 1), count - 1])
    except ValueError:
        return pd.Series([np.nan, np.nan])


def mapprice(v):
    if pd.isnull(v):
        return np.nan
    if " - " in v:
        vals = list(map(float, v.split(" - ")))
        return sum(vals) / len(vals)
    return float(v)


data = pd.read_csv("AmazonProductRating.csv")
data["average_rating"].replace([' out of 5 stars'], '', regex=True, inplace=True)
data["price"].replace([','], '', regex=True, inplace=True)
#data["number_available_in_stock"].replace(['new', 'collectible', 'used'], '', regex=True, inplace=True)

data.drop(["uniq_id", "product_name"], inplace=True, axis=1)

data[["avgprice", "sellercount"]] = data["sellers"].apply(mapseller)

data[["number_available_in_stock", "stock_type"]] = data["number_available_in_stock"].str.extract(r'(\d+)(new|collectible|used)')
data["number_available_in_stock"] = pd.to_numeric(data["number_available_in_stock"], downcast="integer", errors='coerce')
data["number_available_in_stock"]=data["number_available_in_stock"].fillna(data['number_available_in_stock'].mean())
data["stock_type"] = data["stock_type"].fillna(data['stock_type'].mode())


data["bestsellerRank"] = data["product_information"].str.extract(r'Best Sellers Rank ([\d+,?]+) in').replace(',', '',regex=True)
data["bestsellerRank"] = pd.to_numeric(data["bestsellerRank"], downcast="integer", errors='coerce')
data["bestsellerRank"] = data["bestsellerRank"].fillna(data['bestsellerRank'].mean())

data["categoryRank"] = data["product_information"].str.extract(r' #(\d+) ').replace(',', '',regex=True)
data["categoryRank"] = pd.to_numeric(data["categoryRank"], downcast="integer", errors='coerce')
data["categoryRank"] = data["categoryRank"].fillna(data['categoryRank'].mean())

data["LastSubCat"] = data["amazon_category_and_sub_category"].str.extract(r'.*> (.*)$')
#encoding
data = Feature_Encoder(data, ["manufacturer","stock_type","LastSubCat"])

data["price"] = data["price"].apply(mapprice).fillna(data["avgprice"])
data["price"] = data["price"].fillna(data['price'].mean())
data["avgprice"] = data["avgprice"].fillna(data['avgprice'].mean())

data["sellercount"] = data["sellercount"].fillna(data["sellercount"].mean())

data["number_of_reviews"] = pd.to_numeric(data["number_of_reviews"], downcast="integer", errors='coerce')
data["number_of_answered_questions"] = pd.to_numeric(data["number_of_answered_questions"], downcast="integer", errors='coerce')
data["number_of_answered_questions"] = data["number_of_answered_questions"].fillna(data["number_of_answered_questions"].median())

data["average_rating"] = pd.to_numeric(data["average_rating"], downcast="float", errors='coerce')

data = data.dropna(subset=['average_rating', 'number_of_reviews'])



X1 = data.drop([ "sellers", "amazon_category_and_sub_category", "product_information"], inplace=False,
              axis=1)
#X = data.drop(['average_rating', "sellers", "amazon_category_and_sub_category", "product_information"], inplace=False,axis=1)

corr =X1.corr()
#Top 1% Correlation training features with the Value
top_feature = corr.index[abs(corr['average_rating']>0.01)]
#plt.subplots(figsize=(12, 8))
top_corr = X1[top_feature].corr()
#sns.heatmap(top_corr, annot=True)
#plt.show()
X1 = X1[top_feature]
X1=X1.drop('average_rating',axis=1)

#plot for all features
plt.subplots(figsize=(12, 8))
sns.heatmap(data[["price","manufacturer","number_of_reviews","number_of_answered_questions","number_available_in_stock","sellercount","bestsellerRank","categoryRank","avgprice","stock_type","LastSubCat","average_rating"]].corr(), annot=True)
plt.show()

Y = data["average_rating"]


scaler = StandardScaler()
scaler.fit(X1)
X1 = scaler.transform(X1)


X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.30, shuffle=False)
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_train_predicted = poly_model.predict(X_train_poly)
print('Training Mean Square Error', metrics.mean_squared_error(y_train, y_train_predicted))

prediction = poly_model.predict(poly_features.fit_transform(X_test))
print('testing Mean Square Error', metrics.mean_squared_error(y_test, prediction))


# select the best alpha with RidgeCV
alpha_range = 10.**np.arange(-2, 3)
ridgeregcv = RidgeCV(alphas=alpha_range,  scoring='neg_mean_squared_error')
ridgeregcv.fit(X_train_poly, y_train)
y_pred = ridgeregcv.predict(poly_features.fit_transform(X_test))
print(' Rigde Regularization Mean Square test Error ', metrics.mean_squared_error(y_test, y_pred))


print("Intercept: ",poly_model.intercept_)
#print("coefficient of X: \n",poly_model.coef_)


print('R-Squared value: %.2f' % r2_score(y_test, prediction))

plt.plot(X_train_poly,y_train_predicted, "r-", linewidth=2, label="Predictions")
plt.plot(X_train, y_train, "b.",label='Training points')
plt.plot(X_test, y_test, "g.",label='Testing points')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
#plt.show()
import json
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets, metrics, svm, tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from Pre_processing import *


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

                hold = str(hold).replace(',', '')
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

def preProcessing(FileName) :
    data = pd.read_csv(FileName)
    data["price"].replace([','], '', regex=True, inplace=True)
    data.drop(["uniq_id", "product_name"], inplace=True, axis=1)
    data[["avgprice", "sellercount"]] = data["sellers"].apply(mapseller)
    data[["number_available_in_stock", "stock_type"]] = data["number_available_in_stock"].str.extract(
    r'(\d+)(new|collectible|used)')
    data["number_available_in_stock"] = pd.to_numeric(data["number_available_in_stock"], downcast="integer",
                                                  errors='coerce')
    data["number_available_in_stock"] = data["number_available_in_stock"].fillna(data['number_available_in_stock'].mean())
    data["stock_type"] = data["stock_type"].fillna(data['stock_type'].mode())

    data["bestsellerRank"] = data["product_information"].str.extract(r'Best Sellers Rank ([\d+,?]+) in').replace(',', '',
                                                                                                             regex=True)
    data["bestsellerRank"] = pd.to_numeric(data["bestsellerRank"], downcast="integer", errors='coerce')
    data["bestsellerRank"] = data["bestsellerRank"].fillna(data['bestsellerRank'].mean())

    data["categoryRank"] = data["product_information"].str.extract(r' #(\d+) ').replace(',', '', regex=True)
    data["categoryRank"] = pd.to_numeric(data["categoryRank"], downcast="integer", errors='coerce')
    data["categoryRank"] = data["categoryRank"].fillna(data['categoryRank'].mean())

    data["LastSubCat"] = data["amazon_category_and_sub_category"].str.extract(r'.*> (.*)$')

    data = Feature_Encoder(data, ["manufacturer", "stock_type", "LastSubCat", "ProductGrade"])

    data["price"] = data["price"].apply(mapprice).fillna(data["avgprice"])
    data["price"] = data["price"].fillna(data['price'].mean())
    data["avgprice"] = data["avgprice"].fillna(data['avgprice'].mean())

    data["sellercount"] = data["sellercount"].fillna(data["sellercount"].mean())
    data["number_of_reviews"] = pd.to_numeric(data["number_of_reviews"], downcast="integer", errors='coerce')
    data["number_of_reviews"] = data["number_of_reviews"].fillna(data["number_of_reviews"].median())
    data["number_of_answered_questions"] = pd.to_numeric(data["number_of_answered_questions"], downcast="integer",errors='coerce')
    data["number_of_answered_questions"] = data["number_of_answered_questions"].fillna(data["number_of_answered_questions"].median())
    X1 = data.drop(["sellers", "amazon_category_and_sub_category", "product_information"], inplace=False,
               axis=1)

    corr = X1.corr()
    top_feature = corr.index[abs(corr['ProductGrade']) > 0.04]
    top_corr = X1[top_feature].corr()
    X1 = X1[top_feature]
    X1 = X1.drop('ProductGrade', axis=1)
    return X1,data

import matplotlib
matplotlib.use('webagg')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature

matplotlib.rc('figure', figsize=(8, 5))
np.random.seed(200)

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

# Exploration of the OkCupid dataset
try:
    df = pd.read_pickle('cached_dataframe.pkl')
    print("Data Loaded from pickle")
except Exception:
    # df = pd.read_csv("profiles.csv", nrows =200)
    df = pd.read_csv("profiles.csv")
    print("Data Loaded from csv")




    # print('Dataframe columns:')
    # print(df.columns)

    # print('Number of records:')
    # print(len(df))

    # print('First 5 rows:')
    # print(df.head())

    # print('Full print of 1 row:')
    # print(df.iloc[1, :])


    # print('Values of `job` column:')
    # print(df.job.value_counts())


    # print('Values of `status` column:')
    # print(df.status.value_counts())


    # print('Values of `offspring` column:')
    # print(df.offspring.value_counts())

    # print('Values of `speaks` column:')
    # print(df.speaks.value_counts())

    # print('Values of `body_type` column:')
    # print(df.body_type.value_counts())

    # print('Values of `sex` column:')
    # print(df.sex.value_counts())

    # print('Values of `religion` column:')
    # print(df.religion.value_counts())

    # print('Data types of the columns of the dataset:')
    # print(df.dtypes)

    # print('Values of `speaks` column:')
    # print(df.speaks.value_counts())

    # print('Show the columns that have NAN values:')
    # print(df.isna().sum())

    #df = df.dropna(subset=['age', 'height', 'income'])
    # plot by age and gender

    # plt.hist(df[df["sex"] == "m"].age,
    #          bins=20, orientation='horizontal', color=["mediumpurple"], label=["Male"], alpha=0.6)
    # plt.hist(df[df["sex"] == "f"].age,
    #          bins=20, orientation='horizontal', color=["limegreen"], label=["Female"], alpha=0.6)
    # plt.ylabel("Age (years)")
    # plt.xlabel("Frequency")
    # plt.ylim(16, 80)

    # plt.legend()
    # plt.show()

    # # plot by income and gender
    # plt.hist(df[df["sex"] == "m"].income,
    #          bins=20, orientation='horizontal', color=["mediumpurple"], label=["Male"], alpha=0.6)
    # plt.hist(df[df["sex"] == "f"].income,
    #          bins=20, orientation='horizontal', color=["limegreen"], label=["Female"], alpha=0.6)
    # plt.ylabel("Income per year")
    # plt.xlabel("Frequency")
    # plt.legend()
    # plt.show()

    # plot by Religion and gender
    # religioncount = df.religion.value_counts()
    # plt.barh(religioncount.index, religioncount.values)
    # plt.subplots_adjust(left=0.4)
    # plt.show()

    # # plot by education
    # educationcount = df.education.value_counts()
    # plt.barh(educationcount.index, educationcount.values)
    # plt.subplots_adjust(left=0.4)
    # plt.show()


    # augment speaks data
    df = pd.concat([df, (df['speaks'].str.get_dummies(sep=', ').add_prefix('speaks_')) ], axis=1)

    # augment ethnicity data
    df = pd.concat([df, (df['ethnicity'].str.get_dummies(sep=', ').add_prefix('ethnicity_')) ], axis=1)

    # augment based on the essays
    essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
    all_essays = df[essay_cols].replace(np.nan, '', regex=True)
    all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

    df["essay_len"] = all_essays.apply(lambda x: len(x))

    # sentiment analisys on the essays
    df[essay_cols].fillna("", inplace=True  )
    analyser = SentimentIntensityAnalyzer()
    df[essay_cols] = df[essay_cols].astype(str)
    df["essay0_sentiment_score"] = df["essay0"].map(lambda x: analyser.polarity_scores(x)["compound"])
    df["essay1_sentiment_score"] = df["essay1"].map(lambda x: analyser.polarity_scores(x)["compound"])
    df["essay2_sentiment_score"] = df["essay2"].map(lambda x: analyser.polarity_scores(x)["compound"])
    df["essay3_sentiment_score"] = df["essay3"].map(lambda x: analyser.polarity_scores(x)["compound"])
    df["essay4_sentiment_score"] = df["essay4"].map(lambda x: analyser.polarity_scores(x)["compound"])
    df["essay5_sentiment_score"] = df["essay5"].map(lambda x: analyser.polarity_scores(x)["compound"])
    df["essay6_sentiment_score"] = df["essay6"].map(lambda x: analyser.polarity_scores(x)["compound"])
    df["essay7_sentiment_score"] = df["essay7"].map(lambda x: analyser.polarity_scores(x)["compound"])
    df["essay8_sentiment_score"] = df["essay8"].map(lambda x: analyser.polarity_scores(x)["compound"])
    df["essay9_sentiment_score"] = df["essay9"].map(lambda x: analyser.polarity_scores(x)["compound"])
    print("Sentiment analysis Data Augmented")

    # augment categorical data: diet, body_type,  'job', 'sex'
    df = pd.concat([df, (df['diet'].str.get_dummies(sep=', ').add_prefix('diet_')) ], axis=1)
    df = pd.concat([df, (df['body_type'].str.get_dummies(sep=', ').add_prefix('body_type_')) ], axis=1)
    df = pd.concat([df, (df['job'].str.get_dummies(sep=', ').add_prefix('job_')) ], axis=1)
    df = pd.concat([df, (df['sex'].str.get_dummies(sep=', ').add_prefix('sex_')) ], axis=1)

    print("Categorical Data Augmented")
    
    # map 'drinks' into codes
    drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
    df["drinks_code"] = df.drinks.map(drink_mapping)

    # map 'drugs' into codes
    drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
    df["drugs_code"] = df.drugs.map(drugs_mapping)

    # map 'smokes' into codes
    smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "trying to quit": 3, "yes": 4}
    df["smokes_code"] = df.smokes.map(smokes_mapping)

    # map 'pet' into codes
    pets_mapping = {"likes dogs and likes cats" : 2, "likes dogs" : 1, "likes dogs and has cats":3, "has dogs" : 2, "has dogs and likes cats" : 3, "likes dogs and dislikes cats": 0, "has dogs and has cats": 4, "has cats": 2, "likes cats": 1, "has dogs and dislikes cats": 1, "dislikes dogs and likes cats": 0, "dislikes dogs and dislikes cats": -2, "dislikes cats": -1, "dislikes dogs and has cats": 1, "dislikes dogs":-1 }
    df["pets_code"] = df.pets.map(pets_mapping)

    # map 'education' into codes
    education_mapping = {"graduated from college/university": 3,"graduated from masters program": 3,"graduated from two-year college": 3,"graduated from high school": 1,"graduated from ph.d program": 5,"graduated from law school": 4,"graduated from space camp": 4,"graduated from med school": 4,"college/university": 4,"working on college/university": 3,"working on masters program": 3,"working on two-year college": 3,"working on ph.d program": 4,"working on space camp": 3,"working on law school": 3,"two-year college": 3,"working on med school": 3,"masters program": 3,"high school": 1,"working on high school": 1,"space camp": 3,"ph.d program": 4,"law school": 3,"med school": 3,"dropped out of college/university": 2,"dropped out of space camp": 2,"dropped out of two-year college": 2,"dropped out of masters program": 2,"dropped out of ph.d program": 2,"dropped out of high school": 0,"dropped out of law school": 2,"dropped out of med school": 2}
    df["education_code"] = df.education.map(education_mapping)

    print("Ordinal Data Augmented")

    df.to_pickle('cached_dataframe.pkl') # will be stored in current directory

print("len(df):", len(df))
print( df.head())

# Normalize data
# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# feature_data = df.select_dtypes(include=numerics)

# x = feature_data.values
# min_max_scaler = MinMaxScaler()

# x_scaled = min_max_scaler.fit_transform(x)

# print("Data Normalized")


# feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
# feature_data.dropna(inplace=True) # this will drop only 8% of data, so I decided to contine


# print("Drop NAN")

#### Q1: Let's find how many clusters there are:
# print("Kmeans Analysis start")

# x = []
# y = []

# for i in range(1, 20):
#     print(i)
#     classifier = KMeans(i)
#     classifier.fit(feature_data)
#     x.append(i)
#     y.append(classifier.inertia_)

# plt.scatter(x, y)
# plt.xlabel("How many clusters")
# plt.ylabel("Score (inertia)")
# plt.show()


# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)

# print("correlations of sex_m")
# print (feature_data.corr()['sex_m'].sort_values())

# print("correlations of sex_f")
# print (feature_data.corr()['sex_f'].sort_values())

# print("correlations of income")
# print (feature_data.corr()['income'].sort_values())

# top_correlations = get_top_abs_correlations(feature_data, 200)
# print top_correlations


#### Q2: Can I predict 'sex' by sentiment analysis score?
data = df[["essay0_sentiment_score", "essay1_sentiment_score", "essay2_sentiment_score", "essay3_sentiment_score", "essay4_sentiment_score", "essay5_sentiment_score", "essay6_sentiment_score", "essay7_sentiment_score", "essay8_sentiment_score", "essay9_sentiment_score"]]
labels = df[["sex"]].values.ravel()

training_data, validation_data, training_labels, validation_labels = train_test_split(data, labels, test_size=0.2, random_state=100)

# x = []
# y = []

# for i in range(40, 50): # Best result were at k = 47
#     print i
#     classifier = KNeighborsClassifier(i)
#     classifier.fit(training_data, training_labels)
#     result = classifier.score(validation_data, validation_labels)
#     x.append(i)
#     y.append(result)


# plt.scatter(x, y)
# plt.xlabel("K")
# plt.ylabel("Score")
# plt.show()

# classifier = KNeighborsClassifier(47)
# classifier.fit(training_data, training_labels)

# # = classifier.predict(validation_data)
# probas_=classifier.predict_proba( validation_data)
# average_precision = average_precision_score(validation_labels, probas_[:, 1], average="micro", pos_label="m")

# precision, recall, _ = precision_recall_curve(validation_labels, probas_[:, 1], pos_label="m")

# # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
# step_kwargs = ({'step': 'post'}
#                if 'step' in signature(plt.fill_between).parameters
#                else {})
# plt.step(recall, precision, color='b', alpha=0.2,
#          where='post')
# plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
#           average_precision))

# plt.show()



# classifier = SVC()
# classifier.fit(training_data, training_labels)
# print(classifier.score(validation_data, validation_labels))

# y_score = classifier.decision_function(validation_data)
# average_precision = average_precision_score(validation_labels, y_score, average="micro", pos_label="m")

# precision, recall, _ = precision_recall_curve(validation_labels, y_score, pos_label="m")

# # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
# step_kwargs = ({'step': 'post'}
#                if 'step' in signature(plt.fill_between).parameters
#                else {})
# plt.step(recall, precision, color='b', alpha=0.2,
#          where='post')
# plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
#           average_precision))

# plt.show()

# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# feature_data = df.select_dtypes(include=numerics)
# feature_data.dropna(inplace=True)

# # data = feature_data[["sex_m", "sex_f", "drinks_code",  "drugs_code",  "smokes_code",  "pets_code",  "education_code"]]
# data = feature_data.loc[:, feature_data.columns != 'income']
# labels = feature_data[["income"]].values.ravel()

# training_data, validation_data, training_labels, validation_labels = train_test_split(data, labels, test_size=0.001, random_state=100)

# lm = LinearRegression()

# model = lm.fit(training_data, training_labels)

# y_predict = lm.predict(validation_data)

# print("Train score:")
# print(lm.score(training_data, training_labels))

# print("Test score:")
# print(lm.score(validation_data, validation_labels))

# plt.scatter(validation_labels, y_predict)

# plt.xlabel("Income: $Y_i$")
# plt.ylabel("Predicted Income: $\hat{Y}_i$")
# plt.title("Actual Income vs Predicted Income")

# plt.show()
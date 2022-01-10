# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model

#  Load data and prepare to train model.
data_path = "./data_scraping/score_data.csv"
df = pd.read_csv(data_path, index_col=0)

df["team1"].unique()
all_teams = pd.Series(np.concatenate((df["team1"], df["team2"])))
teams = all_teams.unique()

enc_team = OneHotEncoder().fit(teams.reshape(-1, 1))
enc_month = OneHotEncoder().fit(df["month"].values.reshape(-1, 1))
# enc_hours = OneHotEncoder().fit(df["hours"].values.reshape(-1, 1))

team_home = enc_team.transform(df["team1"].values.reshape(-1, 1)).toarray()
team_away = enc_team.transform(df["team2"].values.reshape(-1, 1)).toarray()
months = enc_month.transform(df["month"].values.reshape(-1, 1)).toarray()
# hours = enc_hours.transform(df["hours"].values.reshape(-1,1)).toarray()

X = np.concatenate((team_home, team_away, months), axis=1)
Y = df[["team1_score", "team2_score"]].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)


# %% Section for sklearn models Lasso , ElasticNet, ridge.

def model_fit(model, X_train, y_train, X_test, y_test):
    """train and return model"""
    model.fit(X_train, y_train)
    model_score = model.score(X_test, y_test)
    print("R^2 Test: ", float(model_score))

    return model


datasets = {"X_train": X_train, "y_train": Y_train,
            "X_test": X_test, "y_test": Y_test}

lasso = model_fit(linear_model.Lasso(alpha=.001), **datasets)
#  Lasso is l1 regularization. Implementation gives statistic error.
print("Lasso training score: ", lasso.score(X_train, Y_train))
print("Lasso test score: ", lasso.score(X_test, Y_test))

enet = model_fit(linear_model.ElasticNet(alpha=0.001), **datasets)
#  ElasticNet implementation score for R^2:  0.5426604186969508.
print("ElasticNet training score: ", enet.score(X_train, Y_train))
print("ElasticNet test score: ", enet.score(X_test, Y_test))

ridge = model_fit(linear_model.Ridge(alpha=2.), **datasets)
#  Ridge reg. is l2 regularization. Score for R^2, TRAIN: 0.6643742064009975;  TEST :0.510117603828905
print("Ridge training score: ", ridge.score(X_train, Y_train))
print("Ridge test score: ", ridge.score(X_test, Y_test))

# %% Section for Tensorflow (ANN) model implementation.
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

ann = tf.keras.Sequential([
    tf.keras.layers.Dense(400, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(400, activation="relu"),
    tf.keras.layers.Dense(2, kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001))
])

ann.summary()
num_epochs = 25
BATCH_SIZE = 512
# shuffle(X_train.shape[0]).
dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(BATCH_SIZE)
valset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(X_test.shape[0])

optimizer = tf.keras.optimizers.Adam(0.001)
ann.compile(loss='mse', optimizer=optimizer)
history = ann.fit(dataset, epochs=num_epochs, validation_data=valset)


def calculate_r2(X, Y, model):
    """Calculates R^2 score for given dependent, predicted variables and model"""
    rss = np.sum((Y - model.predict(X)) ** 2)
    tss = np.sum((Y - np.mean(Y, axis=0)) ** 2)

    return 1 - (rss / tss)


print("Test set R^2 score: ", calculate_r2(X_test, Y_test, ann))
print("Trainset R^2 score: ", calculate_r2(X_train, Y_train, ann))

#  %% Calculate models training and test results.

models = {"Lasso Regression": lasso, "Elastic Net": enet,
          "Ridge Regression": ridge, "ANN": ann}

models_scores = {}
for key in models:
    r2_train = calculate_r2(X_train, Y_train, models[key])
    r2_test = calculate_r2(X_test, Y_test, models[key])
    mse_train = np.mean(np.square(Y_train - models[key].predict(X_train)))
    mse_test = np.mean(np.square(Y_test - models[key].predict(X_test)))
    models_scores[key] = {"R^2 Train": r2_train, "R^2 Test": r2_test,
                          "MSE Train": mse_train, "MSE Test": mse_test}

labels = ["Lasso Regression", "Elastic Net", "Ridge Regression", "ANN"]
score_lister = lambda s_name: [np.around(models_scores[labels[i]][s_name], decimals=2)
                               for i in range(len(labels))]

# %% Visualize results.

mpl.rc('axes', titlesize=24, labelsize=22)
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16


def plot_score_hist(labels, training_score, test_score, score_name):
    """Output demonstration plot as histrogram."""
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(figsize=(16, 9))
    rects1 = ax.bar(x - width / 2, training_score, width, label='Training Set')
    rects2 = ax.bar(x + width / 2, test_score, width, label='Test Set')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(score_name)
    ax.set_title(f'{score_name} Scores by Models and Data Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=20)
    fig.tight_layout()
    plt.show()


training_scorer2 = score_lister('R^2 Train')
test_scorer2 = score_lister('R^2 Test')
plot_score_hist(labels, training_scorer2, test_scorer2, "R^2")

training_scoremse = score_lister('MSE Train')
test_scoremse = score_lister('MSE Test')
plot_score_hist(labels, training_scoremse, test_scoremse, "MSE")

# APA Style table
rows = [key for key in models_scores["ANN"]]
cell_text = np.concatenate((np.array([training_scorer2]), np.array([test_scorer2]),
                            np.array([training_scoremse]), np.array([test_scoremse])),
                           axis=0)
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_title("Scores", weight='bold', size=24, color='k')
ax.axis('tight')
fig.patch.set_visible(False)
ax.axis('off')
table = ax.table(cellText=cell_text,
                 cellLoc="center",
                 rowLabels=rows,
                 colLabels=labels,
                 rowLoc='center',
                 loc='center',
                 edges='BT')
table.scale(1, 8)
table.set_fontsize(24)
fig.tight_layout()

plt.show()

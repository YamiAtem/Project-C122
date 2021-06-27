# %% [markdown]
# # Project C122
# %% [markdown]
# ## Getting Data

# %%
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy
import pandas

x = numpy.load('image.npz')['arr_0']
y = pandas.read_csv(
    "https://raw.githubusercontent.com/whitehatjr/datasets/master/C%20122-123/labels.csv")["labels"]

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', "K", "L",
           "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
n_classes = len(classes)

# %% [markdown]
# ## Train Test Split and Scaling

# %%

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=9, train_size=7500, test_size=2500)
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0


# %%

samples_per_class = 5
figure = plt.figure(figsize=(n_classes*2, (1+samples_per_class*2)))

idx_cls = 0
for cls in classes:
    idxs = numpy.flatnonzero(y == cls)
    idxs = numpy.random.choice(idxs, samples_per_class, replace=False)
    i = 0
    for idx in idxs:
        plt_idx = i * n_classes + idx_cls + 1
        p = plt.subplot(samples_per_class, n_classes, plt_idx)
        p = seaborn.heatmap(numpy.reshape(x[idx], (22, 30)), cmap=plt.cm.gray,
                            xticklabels=False, yticklabels=False, cbar=False)
        p = plt.axis('off')
        i += 1
    idx_cls += 1

# %% [markdown]
# ## Logistic Regression

# %%

log_reg = LogisticRegression(solver='saga', multi_class='multinomial')
log_reg.fit(x_train_scaled, y_train)


# %%

prediction = log_reg.predict(x_test_scaled)

accuracy = accuracy_score(y_test, prediction)
print(accuracy)


# %%
confusion_matrix = pandas.crosstab(y_test, prediction, rownames=[
                                   'Actual'], colnames=['Predicted'])

p = plt.figure(figsize=(10, 10))
p = seaborn.heatmap(confusion_matrix, annot=True, fmt="d", cbar=False)

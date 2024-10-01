
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784')

mnist.data.shape

def showimage(dframe, index):    
    some_digit = dframe.to_numpy()[index]
    some_digit_image = some_digit.reshape(28,28)

    plt.imshow(some_digit_image,cmap="binary")
    plt.axis("off")
    plt.show()

showimage(mnist.data, 0)

train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)

type(train_img)

test_img_copy = test_img.copy()

showimage(test_img_copy, 2)

scaler = StandardScaler()

scaler.fit(train_img)

train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

pca = PCA(.95)

pca.fit(train_img)

print(pca.n_components_)

train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

logisticRegr = LogisticRegression(solver = 'lbfgs', max_iter=10000)

logisticRegr.fit(train_img, train_lbl)

logisticRegr.predict(test_img[0].reshape(1,-1))

showimage(test_img_copy, 0)

logisticRegr.predict(test_img[1].reshape(1,-1))

showimage(test_img_copy, 1)

showimage(test_img_copy, 9900)

logisticRegr.predict(test_img[9900].reshape(1,-1))

showimage(test_img_copy, 9999)

logisticRegr.predict(test_img[9999].reshape(1,-1))

logisticRegr.score(test_img, test_lbl)
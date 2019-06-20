---
title: "Deep Bayesian Active Learning on MNIST"
layout: home
classes: wide
date: 2019-06-19
tags:
  - Active Learning
  - Deep Learning
  - Keras
  - Python
header:
    teaser: "assets/images/MNIST.jpeg"
excerpt: "Learning to classify digits with as few data as possible"
---

This is an implementation of the paper Deep Bayesian Active Learning with Image Data using keras and modAL. [modAL](https://modal-python.readthedocs.io/en/latest/) is an active learning framework for Python3, designed with modularity, flexibility and extensibility in mind. Built on top of scikit-learn, it allows you to rapidly create active learning workflows with nearly complete freedom. What is more, you can easily replace parts with your custom built solutions, allowing you to design novel algorithms with ease.

## Active Learning

In this notebook, we are concerned with pool-based Active Learning. In this setting, we have a large amount of unlabelled data and a small initial labelled training set and we want to choose what data should be labelled next.

To do so, there are several query strategies. In this notebook, we will be using uncertainty sampling: the data chosen to be annotated is the one that maximizes an uncertainty criterion (entropy, gini index, variation ratios ...).

## Dropout-Based Bayesian Deep Neural Networks

In this Notebook, we will select the data from the unlabelled pool that maximizes the uncertainty of our model. But the model we will be using will be a Bayesian Deep Neural Network.

Unlike Traditional Deep Learning, where we are looking for the set of weights that maximizes the likelihood of the data (MLE), in bayesian deep learning we are looking for the posterior distribution over the weights and the prediction is then obtained by marginalizing out the weights. As a result, Bayesian models are less prone to overfitting. But unfortunately for big deep models, the posterior distribution is intractable, and we need approximations.

In 2015, [Gal and Ghahramani](https://arxiv.org/pdf/1506.02142.pdf) showed that deep models with dropout layers can be viewed as a lightweight bayesian approximation. The prior and posterior distributions are simply Bernoulli distributions (0 or the learned value). And the predictions can be cheaply obtained at test time by performing Monte Carlo integrations with dropout layers activated.

So in a nutshell, Dropout-based Bayesian Neural Nets are simply Neural Nets with Dropout layers activated at test time.


```python
import keras
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasClassifier
from modAL.models import ActiveLearner
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
```

    Using TensorFlow backend.



```python
def create_keras_model():
    model = Sequential()
    model.add(Conv2D(32, (4, 4), activation='relu'))
    model.add(Conv2D(32, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model
```

### create the classifier


```python
classifier = KerasClassifier(create_keras_model)
```

### read training data


```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

### preprocessing


```python
X_train = X_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32') / 255.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
```

### initial labelled data


```python
initial_idx = np.array([],dtype=np.int)
for i in range(10):
    idx = np.random.choice(np.where(y_train[:,i]==1)[0], size=2, replace=False)
    initial_idx = np.concatenate((initial_idx, idx))

X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]
```

### initial unlabelled pool


```python
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)
```

### Query Strategies


```python
def uniform(learner, X, n_instances=1):
    query_idx = np.random.choice(range(len(X)), size=n_instances, replace=False)
    return query_idx, X[query_idx]

def max_entropy(learner, X, n_instances=1, T=100):
    random_subset = np.random.choice(X.shape[0], 2000, replace=False)
    MC_output = K.function([learner.estimator.model.layers[0].input, K.learning_phase()],
                           [learner.estimator.model.layers[-1].output])
    learning_phase = True
    MC_samples = [MC_output([X[random_subset], learning_phase])[0] for _ in range(T)]
    MC_samples = np.array(MC_samples)  # [#samples x batch size x #classes]
    expected_p = np.mean(MC_samples, axis=0)
    acquisition = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size]
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]


```

### Active Learning Procedure


```python
def active_learning_procedure(query_strategy,
                              X_test,
                              y_test,
                              X_pool,
                              y_pool,
                              X_initial,
                              y_initial,
                              estimator,
                              epochs=50,
                              batch_size=128,
                              n_queries=100,
                              n_instances=10,
                              verbose=0):
    learner = ActiveLearner(estimator=estimator,
                            X_training=X_initial,
                            y_training=y_initial,
                            query_strategy=query_strategy,
                            verbose=verbose
                           )
    perf_hist = [learner.score(X_test, y_test, verbose=verbose)]
    for index in range(n_queries):
        query_idx, query_instance = learner.query(X_pool, n_instances)
        learner.teach(X_pool[query_idx], y_pool[query_idx], epochs=epochs, batch_size=batch_size, verbose=verbose)
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
        model_accuracy = learner.score(X_test, y_test, verbose=0)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        perf_hist.append(model_accuracy)
    return perf_hist
```


```python
estimator = KerasClassifier(create_keras_model)
entropy_perf_hist = active_learning_procedure(max_entropy,
                                              X_test,
                                              y_test,
                                              X_pool,
                                              y_pool,
                                              X_initial,
                                              y_initial,
                                              estimator,)
```


```python
estimator = KerasClassifier(create_keras_model)
uniform_perf_hist = active_learning_procedure(uniform,
                                              X_test,
                                              y_test,
                                              X_pool,
                                              y_pool,
                                              X_initial,
                                              y_initial,
                                              estimator,)
```

    Accuracy after query 1: 0.5968
    Accuracy after query 2: 0.6626
    Accuracy after query 3: 0.7010
    Accuracy after query 4: 0.7224
    Accuracy after query 5: 0.7347
    Accuracy after query 6: 0.7800
    Accuracy after query 7: 0.8014
    Accuracy after query 8: 0.8326
    Accuracy after query 9: 0.8146
    Accuracy after query 10: 0.8203
    Accuracy after query 11: 0.8039
    Accuracy after query 12: 0.8138
    Accuracy after query 13: 0.8527
    Accuracy after query 14: 0.8621
    Accuracy after query 15: 0.8630
    Accuracy after query 16: 0.8694
    Accuracy after query 17: 0.8718
    Accuracy after query 18: 0.8849
    Accuracy after query 19: 0.8729
    Accuracy after query 20: 0.8871
    Accuracy after query 21: 0.8804
    Accuracy after query 22: 0.8879
    Accuracy after query 23: 0.8832
    Accuracy after query 24: 0.8954
    Accuracy after query 25: 0.8948
    Accuracy after query 26: 0.9103
    Accuracy after query 27: 0.9148
    Accuracy after query 28: 0.9122
    Accuracy after query 29: 0.9134
    Accuracy after query 30: 0.9153
    Accuracy after query 31: 0.9201
    Accuracy after query 32: 0.9107
    Accuracy after query 33: 0.9203
    Accuracy after query 34: 0.9195
    Accuracy after query 35: 0.9283
    Accuracy after query 36: 0.9185
    Accuracy after query 37: 0.9257
    Accuracy after query 38: 0.9274
    Accuracy after query 39: 0.9250
    Accuracy after query 40: 0.9296
    Accuracy after query 41: 0.9290
    Accuracy after query 42: 0.9287
    Accuracy after query 43: 0.9390
    Accuracy after query 44: 0.9310
    Accuracy after query 45: 0.9358
    Accuracy after query 46: 0.9370
    Accuracy after query 47: 0.9348
    Accuracy after query 48: 0.9286
    Accuracy after query 49: 0.9403



```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
```

# coding=utf-8
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import cPickle as pickle
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from keras.optimizers import Adadelta, SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import EigenvalueRegularizer
from sklearn.svm import SVC

# x_train = pickle.load(open('x_train.pkl', 'r'))
x_test = pickle.load(open('data/x_test.pkl', 'r'))
y_train = pickle.load(open('data/sub_y_train.pkl', 'r'))
y_test = pickle.load(open('data/y_test.pkl', 'r'))
train_vecs = pickle.load(open('data/sub_trainvecs.pkl', 'r'))
print 'Data loaded...'

n_dim = 200

tr_dict = Word2Vec.load('data/Dictionary')

# tr_dict = Word2Vec(size=n_dim, min_count=10)
# tr_dict.build_vocab(x_train)
# tr_dict.train(x_train)
# tr_dict.save('selftrain_dict')
print 'Model trained...'


# Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    error = 0
    for word in text:
        try:
            vec += tr_dict[word].reshape((1, size))
            count += 1.
        except KeyError:
            error += 1
            # print error
            continue
    if count != 0:
        vec /= count
    return vec


# Build test tweet vectors then scale
test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs = scale(test_vecs)
print 'Test vectorized...'

# LR

lr = SGDClassifier(loss='log', penalty='l1')
# print y_train[:400000]
print sum(y_train)
lr.fit(train_vecs[:320000], y_train[:320000])
print 'Test Accuracy: %.2f' % lr.score(test_vecs[:], y_test[:])

with open('data/LogisticRegression.pkl', 'w') as outfile:
    pickle.dump(lr, outfile, -1)

# Create ROC curve

pred_probas = lr.predict_proba(test_vecs)[:, 1]
print y_test[:100]

fig1 = plt.figure('fig1')

fpr, tpr, _ = roc_curve(y_test, pred_probas)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')

plt.show()


# NN

# topmodel = Sequential()
# topmodel.add(Dense(100, input_shape=(n_dim,), activation='tanh', W_regularizer=EigenvalueRegularizer(10)))
# topmodel.add(Dense(25, activation='tanh', W_regularizer=EigenvalueRegularizer(10)))
# topmodel.add(Dense(2, activation='softmax', W_regularizer=EigenvalueRegularizer(10)))
# sgd = SGD(lr=0.00005, momentum=0.0, decay=0.0, nesterov=False)
# topmodel.compile(loss="binary_crossentropy",
#               optimizer=sgd,
#               metrics=['mean_squared_logarithmic_error', 'accuracy'])
#
#
# enc = OneHotEncoder()
# y_train_onehot = np.reshape(y_train, (len(y_train), 1))
#
# enc.fit(y_train_onehot)
# y_train_onehot = np.asarray(enc.transform(y_train_onehot).toarray())
# # print y_train[0], y_train_onehot[0]
#
#
# topmodel.fit(x=train_vecs, y=y_train_onehot,
#           batch_size=200, nb_epoch=10,
#           shuffle=True,validation_split=0.2)
#
# pred = topmodel.predict(test_vecs, batch_size=30)
#
# score = []
# for i, j in pred:
#     if i > j:
#         score.append(0)
#     else:
#         score.append(1)
#
# print score[:10]
# print y_test[:10]
# score = [score[i] == y_test[i] for i in range(len(score))]
#
# print np.sum(score) / (1.0 * len(score))
#
# fig2 = plt.figure('fig2')
# fpr,tpr,_ = roc_curve(y_test, pred[:, 1])
# roc_auc = auc(fpr,tpr)
# plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.legend(loc='lower right')
#
# plt.show()


# SVW

# print 'Model fitting...'
# svc = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
#
#
# svc.fit(train_vecs, y_train)
#
# print 'Model fitted...'
# svc.predict_proba(test_vecs)
# print 'Test Accuracy: %.2f'%svc.score(test_vecs, y_test)
#
# with open('selftrain_svc.pkl','w') as outfile:
#     pickle.dump(svc, outfile, -1)
#
# #Create ROC curve
#
# pred_probas = svc.predict_proba(test_vecs)[:,1]
#
# fig3 = plt.figure('fig3')
#
# fpr,tpr,_ = roc_curve(y_test, pred_probas)
# roc_auc = auc(fpr,tpr)
# plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.legend(loc='lower right')
#
# plt.show()

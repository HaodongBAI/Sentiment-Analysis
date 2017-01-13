# coding=utf-8
import numpy as np
import gensim
import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import EigenvalueRegularizer
from sklearn.linear_model import SGDClassifier
from gensim.models.doc2vec import LabeledSentence
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
import keras.backend as K
import cPickle as pickle


def get_dataset():
    def load_data():
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        unsup = []
        pos_val = 1
        neg_val = 0

        def load(flag='pos', part='train', y_val=0):
            basepath = 'IMDB_data/' + part + '/'
            for reviews in os.listdir(basepath + flag)[:5000]:
                try:
                    assert str.isdigit(reviews[0])
                except Exception:
                    print Exception
                    continue

                if part == 'train':
                    with open(basepath + flag + '/' + reviews, 'r') as infile:
                        x_train.append(infile.readline())
                    assert y_val == pos_val or y_val == neg_val
                    y_train.append(y_val)
                else:
                    with open(basepath + flag + '/' + reviews, 'r') as infile:
                        x_test.append(infile.readline())
                    assert y_val == pos_val or y_val == neg_val
                    y_test.append(y_val)

        load('pos', 'train', pos_val)
        load('neg', 'train', neg_val)
        load('pos', 'test', pos_val)
        load('neg', 'test', neg_val)

        unsup_basepath = 'IMDB_data/train/unsup/'
        for reviews in os.listdir(unsup_basepath)[:2000]:
            try:
                assert str.isdigit(reviews[0])
            except Exception:
                print Exception
                continue

            with open(unsup_basepath + reviews, 'r') as infile:
                unsup.append(infile.readline())

        return x_train, x_test, y_train, y_test, unsup

    print 'Start...'
    x_train, x_test, y_train, y_test, unsup_reviews = load_data()
    print 'Data loaded...'

    def cleanText(corpus):
        punctuation = """.,?!:;(){}[]"""
        corpus = [z.lower().replace('\n', '') for z in corpus]
        corpus = [z.replace('<br />', ' ') for z in corpus]

        # treat punctuation as individual words
        for c in punctuation:
            corpus = [z.replace(c, ' %s ' % c) for z in corpus]
        corpus = [z.split() for z in corpus]
        return corpus

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    unsup_reviews = cleanText(unsup_reviews)
    print 'Data cleaned...'

    def labelizeReviews(reviews, label_type):
        labelized = []
        for i, v in enumerate(reviews):
            label = '%s_%s' % (label_type, i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')
    unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')
    # print len(x_train[0])
    print 'Data labelized...'

    return x_train, x_test, y_train, y_test, unsup_reviews


def run_imdb():
    x_train, x_test, y_train, y_test, unsup_reviews = get_dataset()

    size = 400

    # instantiate our DM and DBOW models
    model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3, dm_concat=1)
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    # build vocab over all reviews
    # model_dm.build_vocab(x_train)
    all_vacob_reviews = [i for i in x_train]
    all_vacob_reviews.extend([i for i in x_test])
    all_vacob_reviews.extend([i for i in unsup_reviews])

    model_dm.build_vocab(all_vacob_reviews)
    model_dbow.build_vocab(all_vacob_reviews)
    print 'Model vocabulary built...'

    # We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.

    all_train_reviews = [i for i in x_train]
    all_train_reviews.extend([i for i in unsup_reviews])
    for epoch in range(5):
        print 'trainset epoch_{}: '.format(epoch),
        perm = np.random.permutation(len(all_train_reviews))
        print perm
        model_dm.train(all_train_reviews)
        model_dbow.train(all_train_reviews)

    print 'Model trained...'

    # Get training set vectors from our models
    def getVecs(model, corpus, size):
        vecs = [np.array(model[z.words[0]]).reshape((1, size)) for z in corpus]
        return np.concatenate(vecs)

    train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)

    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))

    # train over test set
    # x_test = np.array(x_test)

    for epoch in range(10):
        print 'testset epoch_{}: '.format(epoch),
        perm = np.random.permutation(len(x_test))
        print len(perm)
        model_dm.train(x_test)
        model_dbow.train(x_test)

    # Construct vectors for test reviews
    test_vecs_dm = getVecs(model_dm, x_test, size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)

    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)

    try:
        assert hasattr(lr, 'score')
        print 'Test Accuracy: %.2f' % lr.score(test_vecs, y_test)
    except Exception:
        print Exception

    # Create ROC curve
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    pred_probas = lr.predict_proba(test_vecs)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')

    plt.show()


def buildWordVec(text, size, imdb_w2v):
    # error = 0
    vec = np.zeros(size).reshape((1, size))
    count = 0.

    for word in text:
        if word in imdb_w2v:
            # print word,
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.

    # print count

    if count != 0:
        vec /= count
    return vec


def load_model(dim=100):
    weight_path = 'glove.twitter.27B/glove.twitter.27B.{}d.txt'.format(dim)
    model = OrderedDict()
    f = open(weight_path, 'r')
    for l in f.readlines()[:]:
        l.decode('utf-8')
        l = l.replace('\n', '')
        vec = l.split(' ')

        model[vec[0]] = np.asarray([float(i) for i in vec[1:]])
        # print model[vec[0]]
    return model


def load_data():
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def load(flag='pos', part='train', y_val=0):
        basepath = 'trainingandtestdata/' + part + '_' + flag
        f = open(basepath, 'r')
        for l in f.readlines()[:]:
            if part == 'train':
                x_train.append(l.decode('utf-8'))
                assert y_val == pos_val or y_val == neg_val
                y_train.append(y_val)
            else:
                x_test.append(l.decode('utf-8'))
                assert y_val == pos_val or y_val == neg_val
                y_test.append(y_val)

    load('pos', 'train', pos_val)
    load('neg', 'train', neg_val)
    load('pos', 'test', pos_val)
    load('neg', 'test', neg_val)

    def clear(set):
        special = ['<USER>', '<SMILE>', '<URL>', '<LOLFACE>', '<SADFACE>', '<NUMBER>',
                   '<HASHTAG>', '<REPEAT>', '<ELONG>', '<HEART>', '<ALLCAPS>']

        set = [z.replace('< N U M B E R>', ' %s ' % '<NUMBER>') for z in set]

        for c in special:
            set = [z.replace(c, ' %s ' % c) for z in set]

        punctuation = """.,?!:;(){}[]"""
        for c in punctuation:
            set = [z.replace(c, ' %s ' % c) for z in set]

        # set = [[nltk.PorterStemmer().stem_word(w) for w in t.replace('\n', '').replace('  ', ' ').replace('  ', ' ').split(' ')] for t in set]
        set = [t.replace('\n', '').lower().split() for t in set]

        return set

    print 'Before cleaning...'
    x_train = clear(x_train)
    x_test = clear(x_test)
    print 'After cleaning...'

    return x_train, x_test, y_train, y_test


def glove_sentiment():
    print 'Start...'
    n_dim = 200
    pos_val = 1
    neg_val = 0

    x_train, x_test, y_train, y_test = load_data()
    print 'Data loaded...'
    print len(x_train)
    print len(x_test)

    with open('x_train.pkl', 'w') as outfile:
        pickle.dump(x_train, outfile, -1)
    with open('x_test.pkl', 'w') as outfile:
        pickle.dump(x_test, outfile, -1)
    with open('y_train.pkl', 'w') as outfile:
        pickle.dump(y_train, outfile, -1)
    with open('y_test.pkl', 'w') as outfile:
        pickle.dump(y_test, outfile, -1)
    print 'Data saved...'

    # x_train = pickle.load(open('x_train.pkl', 'r'))
    x_test = pickle.load(open('x_test.pkl', 'r'))
    y_train = pickle.load(open('y_train.pkl', 'r'))
    y_test = pickle.load(open('y_test.pkl', 'r'))
    print 'Data loaded...'

    imdb_w2v = pickle.load(open('dict.pkl'))

    print 'Vectors loaded...'

    # Build word vector for training set by using the average value of all word vectors in the tweet, then scale
    def buildWordVector(text, size):
        # error = 0
        vec = np.zeros(size).reshape((1, size))
        count = 0.

        for word in text:
            if word in imdb_w2v:
                # print word,
                vec += imdb_w2v[word].reshape((1, size))
                count += 1.

        # print count

        if count != 0:
            vec /= count
        return vec

    # train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
    # train_vecs = scale(train_vecs)

    # with open('train_vecs.pkl', 'w') as outfile:
    #     pickle.dump(train_vecs, outfile, -1)

    train_vecs = pickle.load(open('train_vecs.pkl', 'r'))
    print 'Train set vectorized...'

    print len(train_vecs[0])
    # K.set_image_dim_ordering('th')
    topmodel = Sequential()
    topmodel.add(Dense(100, input_shape=(n_dim,), activation='tanh', W_regularizer=EigenvalueRegularizer(10)))
    topmodel.add(Dense(25, activation='tanh', W_regularizer=EigenvalueRegularizer(10)))
    topmodel.add(Dense(2, activation='softmax', W_regularizer=EigenvalueRegularizer(10)))

    topmodel.compile(loss="categorical_crossentropy",
                     optimizer='Adadelta',
                     metrics=['mean_squared_logarithmic_error', 'accuracy'])

    enc = OneHotEncoder()
    y_train_onehot = np.reshape(y_train, (len(y_train), 1))
    enc.fit(y_train_onehot)
    y_train_onehot = np.asarray(enc.transform(y_train_onehot).toarray())

    topmodel.fit(x=train_vecs, y=y_train_onehot,
                 batch_size=10000, nb_epoch=5,
                 shuffle=True, validation_split=0.2)

    # Train word2vec on test tweets
    # imdb_w2v.train(x_test)

    # Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
    test_vecs = scale(test_vecs)
    print 'Test set vectorized...'

    with open('test_vecs.pkl', 'w') as outfile:
        pickle.dump(test_vecs, outfile, -1)

    # Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set

    # lr = SGDClassifier(loss='log', penalty='l1')
    # lr.fit(train_vecs, y_train)

    # print 'Test Accuracy: %.2f' % lr.score(test_vecs, y_test)

    # Create ROC curve

    pred = topmodel.predict(test_vecs, batch_size=30)

    print pred[:10]
    score = []
    for i, j in pred:
        if i > j:
            score.append(0)
        else:
            score.append(1)

    print score[:10]
    print y_test[:10]
    score = [score[i] == y_test[i] for i in range(len(score))]

    print np.sum(score) / (1.0 * len(score))

    # fpr, tpr, _ = roc_curve(y_test, pred_probas)
    # roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.legend(loc='lower right')
    #
    # plt.show()

    # with open('dict.pkl','w') as outfile:
    #     pickle.dump(imdb_w2v, outfile, -1)
    topmodel.save('topmodel', True)
    with open('pred.pkl', 'w') as outfile:
        pickle.dump(pred, outfile, -1)

from sys import argv
import numpy as np

def onlineSA():
    from gensim.models.word2vec import Word2Vec
    dict = Word2Vec.load('data/Dictionary')

    import cPickle as pickle
    lr = pickle.load(open('data/LogisticRegression.pkl', 'r'))

    class Analyzer(object):
        def __init__(self, dict, classifier):
            self.dict = dict
            self.classifer = classifier

        def clean(self, text):

            tempin = open('tempin', 'w')
            tempin.write(text)
            tempin.close()
            import os
            tempout = os.popen(r'ruby -n preprocess.rb tempin')

            cleaned_text = tempout.readline()

            special = ['<USER>', '<SMILE>', '<URL>', '<LOLFACE>', '<SADFACE>', '<NUMBER>',
                       '<HASHTAG>', '<REPEAT>', '<ELONG>', '<HEART>', '<ALLCAPS>']

            cleaned_text = cleaned_text.replace('< N U M B E R>', ' %s ' % '<NUMBER>')
            for c in special:
                cleaned_text = cleaned_text.replace(c, ' %s ' % c)

            punctuation = """.,?!:;(){}[]"""
            for c in punctuation:
                cleaned_text = cleaned_text.replace(c, ' %s ' % c)

            cleaned_text = cleaned_text.replace('\n', '').lower().split()
            return cleaned_text

        def accumulate(self, cleaned_text):
            # vec = [0 for i in range(200)]
            vec = np.zeros(shape=(1,200))
            count = 0.
            error = 0
            for word in cleaned_text:
                if word in self.dict:
                    # print word,
                    vec += self.dict[word]
                    # vec = [vec[i]+self.dict[word][i] for i in range(200)]
                    count += 1.
                else:
                    error += 1

            if count != 0:
                vec /= count
                # vec = [vec[i]/count for i in range(200)]
            warning_info = '{} words unfound, {} words accumulated.'.format(error, int(count))
            return vec, warning_info

        def predict(self, text, show_infos=False):
            cleaned_text = self.clean(text)
            sum, warning_info = self.accumulate(cleaned_text)
            pred_prob = self.classifer.predict_proba(sum)[:, 1]

            print 'Cleaned text: ',
            for i in cleaned_text:
                print i,
            print

            if pred_prob > 0.5:
                print 'Prediction: positive {}'.format(pred_prob)
            else:
                print 'Prediction: negtive {}'.format(1 - pred_prob)

            if show_infos:
                print '# Info: ' + warning_info

            return cleaned_text

    return Analyzer(dict, lr)


if __name__ == "__main__":
    ana = onlineSA()
    text = ' '.join(argv[1:], )
    print text
    ana.predict(text, True)
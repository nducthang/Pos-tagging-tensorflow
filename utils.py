from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle
from pyvi import ViTokenizer

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sentences, word = [], []
        for i, line in enumerate(lines):
            if line.isspace():
                sentences.append(word)
                word = []
            else:
                data = line.split('\t')
                data[0] = data[0].replace(' ', '_')
                # if data[3][-1] == '\n':
                #     data[3] = data[3][:-1]
                word.append([data[0], data[2]])
        return sentences


def parse(sentences):
    all_words = []
    all_tags = []
    for sentence in sentences:
        for word, tag in sentence:
            if word not in all_words:
                all_words.append(word)
            if tag not in all_tags:
                all_tags.append(tag)
    all_words.sort()
    all_tags.sort()

    word2idx = {word: idx + 2 for idx, word in enumerate(all_words)}
    word2idx['PAD'] = 0
    word2idx['UNK'] = 1
    tag2idx = {tag: idx + 1 for idx, tag in enumerate(all_tags)}
    tag2idx['PAD'] = 0

    idx2word = {idx: word for word, idx in word2idx.items()}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

    return all_words, all_tags, word2idx, idx2word, tag2idx, idx2tag


def sentence_to_number(sentences, max_length, word2idx, tag2idx):
    x = [[word2idx[word] for word, tag in sent] for sent in sentences]
    y = [[tag2idx[tag] for word, tag in sent] for sent in sentences]
    x_pad = pad_sequences(x, maxlen=max_length, dtype='int32', padding='post', truncating='post', value=word2idx['PAD'])
    y_pad = pad_sequences(y, maxlen=max_length, dtype='int32', padding='post', truncating='post', value=tag2idx['PAD'])
    y_categorical = [to_categorical(idx, num_classes=len(list(tag2idx.keys())), dtype='int32') for idx in y_pad]
    return np.array(x_pad), np.array(y_categorical)


# tạo hàm lấy batch tiếp theo. Khi lấy hết batch của mẫu thì shuffle lại mẫu.
def next_batch(X, Y, batch_size, index=0):
    start = index
    index += batch_size
    if index > len(X):
        perm = np.arange(len(X))
        np.random.shuffle(perm)
        X = X[perm]
        Y = Y[perm]
        start = 0
        index = batch_size
    end = index
    return X[start:end], Y[start:end], index


def save_config(all_words, all_tags, word2idx, idx2word, tag2idx, idx2tag, path_vocab, path_tag, path_word2idx,
                path_idx2word, path_tag2idx, path_idx2tag):
    with open(path_vocab, 'wb') as f:
        pickle.dump(all_words, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path_tag, 'wb') as f:
        pickle.dump(all_tags, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path_word2idx, 'wb') as f:
        pickle.dump(word2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path_idx2word, 'wb') as f:
        pickle.dump(idx2word, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path_tag2idx, 'wb') as f:
        pickle.dump(tag2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path_idx2tag, 'wb') as f:
        pickle.dump(idx2tag, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_config(path_vocab, path_tag, path_word2idx, path_idx2word, path_tag2idx, path_idx2tag):
    with open(path_vocab, 'rb') as f:
        all_words = pickle.load(f)
    with open(path_tag, 'rb') as f:
        all_tags = pickle.load(f)
    with open(path_word2idx, 'rb') as f:
        word2idx = pickle.load(f)
    with open(path_idx2word, 'rb') as f:
        idx2word = pickle.load(f)
    with open(path_tag2idx, 'rb') as f:
        tag2idx = pickle.load(f)
    with open(path_idx2tag, 'rb') as f:
        idx2tag = pickle.load(f)

    return all_words, all_tags, word2idx, idx2word, tag2idx, idx2tag


def tokenizer(list_sentences, all_words, word2idx, max_length):
    sentences = [ViTokenizer.tokenize(text).split(' ') for text in list_sentences]
    # convert to number
    sent_matrix = []
    for sent in sentences:
        sent2num = []
        for word in sent:
            if word in all_words:
                sent2num.append(word2idx[word])
            else:
                sent2num.append(word2idx['UNK'])
        sent_matrix.append(sent2num)
    sent_matrix = pad_sequences(sent_matrix, maxlen=max_length, dtype='int32', padding='post', truncating='post',
                             value=word2idx['PAD'])
    return sentences, np.array(sent_matrix)
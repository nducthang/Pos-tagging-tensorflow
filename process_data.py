from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

path = './data/train.txt'
MAX_LENGTH = 150


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
                data[0] = data[0].lower().replace(' ', '_')
                if data[3][-1] == '\n':
                    data[3] = data[3][:-1]
                word.append([data[0], data[3]])
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

    return word2idx, idx2word, tag2idx, idx2tag


def sentence_to_number(sentences, max_length, word2idx, tag2idx):
    x = [[word2idx[word] for word, tag in sent] for sent in sentences]
    y = [[tag2idx[tag] for word, tag in sent] for sent in sentences]
    x_pad = pad_sequences(x, maxlen=max_length, dtype='int32', padding='post', truncating='post', value=word2idx['PAD'])
    y_pad = pad_sequences(y, maxlen=max_length, dtype='int32', padding='post', truncating='post', value=tag2idx['PAD'])
    y_categorical = [to_categorical(idx, num_classes=len(list(tag2idx.keys())), dtype='int32') for idx in y_pad]
    return np.array(x_pad), np.array(y_categorical)


if __name__ == '__main__':
    sentences = load_data(path)
    word2idx, idx2word, tag2idx, idx2tag = parse(sentences)
    X_train, y_train = sentence_to_number(sentences, MAX_LENGTH, word2idx, tag2idx)

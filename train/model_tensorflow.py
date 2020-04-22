import tensorflow as tf
import utils
from sklearn.model_selection import train_test_split
import numpy as np
import config.config as config

# tf.executing_eagerly()

# parameters
MAX_LENGTH = config.MAX_LENGTH
EMBEDDING_SIZE = config.EMBEDDING_SIZE
NUM_UNIT = config.NUM_UNIT
DROPOUT_RATE = config.DROPOUT_RATE
NUM_STEP = config.NUM_STEP
BATCH_SIZE = config.BATCH_SIZE
DISPLAY_STEP = config.DISPLAY_STEP
SAVE_PATH_CHECKPOINT = config.SAVE_PATH_CHECKPOINT

PATH_TRAIN = config.PATH_TRAIN
PATH_ALLWORDS = config.PATH_ALLWORDS
PATH_ALLTAGS = config.PATH_ALLTAGS
PATH_WORD2IDX = config.PATH_WORD2IDX
PATH_IDX2WORD = config.PATH_IDX2WORD
PATH_TAG2IDX = config.PATH_TAG2IDX
PATH_IDX2TAG = config.PATH_IDX2TAG


class Model(object):
    def __init__(self, vocab_size, num_classes):
        self.VOCAB_SIZE = vocab_size
        self.NUM_CLASSES = num_classes
        self.EMBEDDING_SIZE = EMBEDDING_SIZE
        self.DROPOUT_RATE = DROPOUT_RATE
        self.MAX_LENGTH = MAX_LENGTH
        self.NUM_UNIT = NUM_UNIT
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_STEP = NUM_STEP
        self.DISPLAY_STEP = DISPLAY_STEP
        self.SAVE_PATH_CHECKPOINT = SAVE_PATH_CHECKPOINT

    def build_model(self):
        self.X = tf.placeholder(dtype='int32', shape=[None, self.MAX_LENGTH], name='X')
        self.Y = tf.placeholder(dtype='int32', shape=[None, self.MAX_LENGTH, self.NUM_CLASSES], name='Y')
        self.embedding = self._embedding(self.X)
        self.bilstm_1 = self._bilstm(self.embedding, self.NUM_UNIT)
        self.dropout_1 = self._dropout(self.bilstm_1)
        self.bilstm_2 = self._bilstm(self.dropout_1, 2 * self.NUM_UNIT)
        self.dropout_2 = self._dropout(self.bilstm_2)
        self.logits = self._logits(self.dropout_2)
        self.predict = self._predict(self.logits)
        self.loss = self._loss(self.logits, self.Y)
        self.train = self._train(self.loss)
        self.accuracy = self._accuracy(self.predict, self.Y)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        print("Build model succesfull!")

    def _embedding(self, X):
        with tf.variable_scope('embeding'):
            embed_word = tf.contrib.layers.embed_sequence(X, self.VOCAB_SIZE, self.EMBEDDING_SIZE,
                                                          initializer=tf.random_uniform_initializer(seed=28))
        return embed_word

    def _bilstm(self, layer_pre, num_unit):
        with tf.variable_scope('BiLSTM' + str(num_unit)):
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(num_unit)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(num_unit)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, layer_pre,
                                                                        dtype=tf.float32)
        output = tf.concat([output_fw, output_bw], axis=-1)
        return output

    def _dropout(self, layer_pre):
        if self.DROPOUT_RATE > 0.0:
            dropout = tf.nn.dropout(layer_pre, 1.0 - self.DROPOUT_RATE)
        else:
            dropout = layer_pre
        return dropout

    def _logits(self, layer_pre):
        with tf.variable_scope('logits'):
            logits = tf.layers.dense(layer_pre, self.NUM_CLASSES)
        return logits

    def _predict(self, logits):
        with tf.variable_scope('predict'):
            predict = tf.nn.softmax(logits)
        return predict

    def _loss(self, logits, Y):
        with tf.variable_scope('loss_op'):
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        return loss_op

    def _train(self, loss_op):
        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss_op)
        return train_op

    def _accuracy(self, predict, Y):
        with tf.variable_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(predict, 2), tf.argmax(Y, 2))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy

    def fit(self, X_train, Y_train, X_val, Y_val, save=True):
        min_loss = np.inf
        idx = 0
        for step in range(1, self.NUM_STEP + 1):
            X_batch, Y_batch, idx = utils.next_batch(X_train, Y_train, batch_size=self.BATCH_SIZE, index=idx)
            self.sess.run(self.train, feed_dict={self.X: X_batch, self.Y: Y_batch})
            if step == 1 or step % self.DISPLAY_STEP == 0:
                loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict={self.X: X_val, self.Y: Y_val})
                print("Step: {:4}, Validation loss: {:.8f}, Validation accuracy: {}".format(step, loss, acc))
                if save and loss < min_loss:
                    min_loss = loss
                    self.saver.save(self.sess, save_path=self.SAVE_PATH_CHECKPOINT)
        if save:
            print("Training complete! Session save in ", self.SAVE_PATH_CHECKPOINT)
        else:
            print("Training complete! Don't save session!")

    def predict_batch(self, list_sentences, all_words, word2idx, idx2tag):
        sent_token, sent_matrix = utils.tokenizer(list_sentences, all_words, word2idx, self.MAX_LENGTH)
        predict = self.sess.run(tf.argmax(self.predict, 2), feed_dict={self.X: sent_matrix})
        # convert to tag
        tags = []
        for i in range(len(predict)):
            tag_predict = []
            for j in range(len(sent_token[i])):
                tag_predict.append(idx2tag[predict[i][j]])
            tags.append(tag_predict)
        return sent_token, tags

if __name__ == '__main__':
    print("Đọc dữ liệu...")
    sentences = utils.load_data(PATH_TRAIN)
    all_words, all_tags, word2idx, idx2word, tag2idx, idx2tag = utils.parse(sentences)
    X_train, Y_train = utils.sentence_to_number(sentences, MAX_LENGTH, word2idx, tag2idx)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)
    utils.save_config(all_words, all_tags, word2idx, idx2word, tag2idx, idx2tag, PATH_ALLWORDS, PATH_ALLTAGS,
                      PATH_WORD2IDX, PATH_IDX2WORD, PATH_TAG2IDX, PATH_IDX2TAG)
    print("Lưu tham số thành công! Tiến hành training...")
    VOCAB_SIZE = len(word2idx.items())
    NUM_CLASSES = len(tag2idx.items())
    model = Model(VOCAB_SIZE, NUM_CLASSES)
    model.build_model()
    model.fit(X_train, Y_train, X_val, Y_val, save=True)

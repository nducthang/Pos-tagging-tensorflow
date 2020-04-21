import tensorflow as tf
import process_data
from sklearn.model_selection import train_test_split
import numpy as np

# tf.executing_eagerly()

# parameters
PATH_TRAIN = '../data/train.txt'
PATH_TEST = '../data/test.txt'
MAX_LENGTH = 120
EMBEDDING_SIZE = 100
NUM_UNIT = 64
DROPOUT_RATE = 0.1
NUM_STEP = 1000
BATCH_SIZE = 128
DISPLAY_STEP = 100


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


if __name__ == '__main__':
    # load data
    sentences = process_data.load_data(PATH_TRAIN)
    word2idx, idx2word, tag2idx, idx2tag = process_data.parse(sentences)
    X_train, Y_train = process_data.sentence_to_number(sentences, MAX_LENGTH, word2idx, tag2idx)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)
    print("=" * 50)
    print("Shape X_train={}, Y_train={}, X_test={}, Y_test={}".format(X_train.shape, Y_train.shape, X_val.shape,
                                                                      Y_val.shape))
    print("=" * 50)

    VOCAB_SIZE = len(word2idx.items())
    NUM_CLASSES = len(tag2idx.items())

    """ Xây dựng mạng
    Input:(None, 120)
    Embedding:(None, 120, 100)
    Bi-LSTM 1: (None, 120, 64x2)
    Bi-LSTM 2: (None, 120, 128x2)
    Dense: (None, 120, 10)
    """

    X = tf.placeholder(dtype='int32', shape=[None, MAX_LENGTH], name='X')
    Y = tf.placeholder(dtype='int32', shape=[None, MAX_LENGTH, NUM_CLASSES], name='Y')
    # global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    with tf.variable_scope('embedding'):
        embed_word = tf.contrib.layers.embed_sequence(X, VOCAB_SIZE, EMBEDDING_SIZE,
                                                      initializer=tf.random_uniform_initializer(seed=28))
    with tf.variable_scope("Bi-LSTM-1"):
        lstm_fw_cell_1 = tf.contrib.rnn.LSTMCell(NUM_UNIT)
        lstm_bw_cell_1 = tf.contrib.rnn.LSTMCell(NUM_UNIT)
        (output_fw_1, output_bw_1), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, embed_word,
                                                                        dtype=tf.float32)
    output_1 = tf.concat([output_fw_1, output_bw_1], axis=-1)

    with tf.variable_scope("dropout-1"):
        if DROPOUT_RATE > 0.0:
            dropout_1 = tf.nn.dropout(output_1, 1.0 - DROPOUT_RATE)
        else:
            drop_out_1 = output_1

    with tf.variable_scope("Bi-LSTM-2"):
        lstm_fw_cell_2 = tf.contrib.rnn.LSTMCell(NUM_UNIT * 2)
        lstm_bw_cell_2 = tf.contrib.rnn.LSTMCell(NUM_UNIT * 2)
        (output_fw_2, output_bw_2), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, dropout_1,
                                                                        dtype=tf.float32)
    output_2 = tf.concat([output_fw_2, output_bw_2], axis=-1)

    with tf.variable_scope("dropout-2"):
        if DROPOUT_RATE > 0.0:
            dropout_2 = tf.nn.dropout(output_2, 1.0 - DROPOUT_RATE)
        else:
            drop_out_2 = output_2

    with tf.variable_scope("logit"):
        logits = tf.layers.dense(dropout_2, NUM_CLASSES)

    with tf.variable_scope("predict"):
        predict = tf.nn.softmax(logits)

    # Summary:
    print('=' * 50)
    print("Embedding:", embed_word.get_shape())
    print("Bi-LSTM 1:", output_1.get_shape())
    print("Dropout 1:", dropout_1.get_shape())
    print("Bi-LSTM 2:", output_2.get_shape())
    print("Dropout 2:", dropout_2.get_shape())
    print("Dense:", logits.get_shape())
    print('=' * 50)

    with tf.variable_scope("loss"):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    with tf.variable_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss_op)
        # train_op = optimizer.minimize(loss_op, global_step=global_step)

    with tf.variable_scope("evaluate"):
        correct_pred = tf.equal(tf.argmax(predict, 2), tf.argmax(Y, 2))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        idx = 0
        for step in range(1, NUM_STEP + 1):
            X_batch, Y_batch, idx = next_batch(X_train, Y_train, batch_size=BATCH_SIZE, index=idx)
            sess.run(train_op, feed_dict={X: X_batch, Y: Y_batch})
            min_loss = np.inf
            if step == 1 or step % DISPLAY_STEP == 0:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: X_val, Y: Y_val})
                print("Step: {:4}, Validation loss: {}, Validation accuracy: {}".format(step, loss, acc))
                if loss < min_loss:
                    # print("==> Loss tốt nhất giảm từ {:.10} xuống {:.10}, thực hiện lưu model.".format(min_loss, loss))
                    min_loss = loss
                    saver.save(sess, save_path='./checkpoints/model_tensorflow')

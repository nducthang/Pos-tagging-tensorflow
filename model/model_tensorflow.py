import tensorflow as tf
import process_data

if __name__ == '__main__':
    # parameters
    path = '../data/train.txt'
    MAX_LENGTH = 120
    BATCH_SIZE = 128

    sentences = process_data.load_data(path)
    word2idx, idx2word, tag2idx, idx2tag = process_data.parse(sentences)
    X_train, y_train = process_data.sentence_to_number(sentences, MAX_LENGTH, word2idx, tag2idx)
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    X, Y = iterator.get_next()

    # paramters
    vocab_size = len(word2idx.items())
    num_classes = len(tag2idx.items())
    embedding_size = 50
    n_hidden_gate = 64
    dropout_rate = 0.5

    with tf.variable_scope('embedding'):
        embed_word = tf.contrib.layers.embed_sequence(X, vocab_size, embedding_size,
                                                      initializer=tf.random_normal_initializer(0, 0.01, seed=2))

    with tf.variable_scope('bi-lstm'):
        lstm_fw_cell = tf.contrib.rnn.LSTMCell(n_hidden_gate)
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(n_hidden_gate)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embed_word,
                                                                    dtype=tf.float32)

    output = tf.concat([output_fw, output_bw], axis=-1)

    with tf.variable_scope('bi-lstm-2'):
        lstm_fw_cell_2 = tf.contrib.rnn.LSTMCell(n_hidden_gate * 2)
        lstm_bw_cell_2 = tf.contrib.rnn.LSTMCell(n_hidden_gate * 2)
        (output_fw_2, output_bw_2), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, output,
                                                                        dtype=tf.float32)

    output_2 = tf.concat([output_fw_2, output_bw_2], axis=-1)

    if dropout_rate > 0.0:
        output = tf.nn.dropout(output_2, 1.0 - dropout_rate)

    logits = tf.layers.dense(output_2, num_classes)
    predict = tf.nn.softmax(logits)

    # define loss and optimizer
    #loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    loss_op = tf.losses.mean_squared_error(Y,logits )
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # init variables
    init = tf.global_variables_initializer()
    training_step = 100000
    skip_step = 50
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(init)
        print("Khởi tạo xong biến!")
        total_loss = 0
        for step in range(1, training_step + 1):
            try:
                sess.run(train_op)
                loss_batch, acc = sess.run([loss_op, accuracy])
                total_loss += loss_batch
                if step == 1 or step % skip_step == 0:
                    print("Step = {}, Average Loss = {:.4f}, Acc = {:.3f}".format(step, total_loss / skip_step, acc))
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
        print("Done!")
        sess.run(train_op)
        loss, acc = sess.run([loss_op, accuracy])
        print("Loss = {:.4f}, Acc = {:.3f}".format(loss, acc))

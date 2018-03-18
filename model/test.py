import tensorflow as tf

w = tf.get_variable(
    'test',
    initializer=tf.random_uniform_initializer(-1, 1),
    shape=(2, 10, 5),
    dtype=tf.float32)

l = tf.get_variable(
    'label',
    initializer=tf.random_uniform_initializer(-1, 1),
    shape=(2, 10),
    dtype=tf.float32)

concat = tf.concat(l, -1)
output = tf.reshape(tf.concat(w, 1), [-1, 5])

reshape = tf.reshape(w, (-1, 5))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()

    l, c = sess.run([l, concat])

    o = sess.run(output)

    print(o)
    print(o.shape)


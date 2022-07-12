import tensorflow as tf
hello_constant=tf.constant('Hello World')
with tf.Session() as sess:
    output=sess.run(hello_constant)
    print(output)
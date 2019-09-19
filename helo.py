import tensorflow as tf
message = tf.constant('welcome to the exciting world of Deep Neural Networks!')
with tf.Session() as sess:
	print(sess.run(message).decode())
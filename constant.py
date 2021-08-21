# Define constant variables
import tensorflow as tf

a = tf.range(start=0, limit=64, dtype=tf.float32)

print(a[None, :].shape)
print(tf.reshape(a, (1, -1)).shape)
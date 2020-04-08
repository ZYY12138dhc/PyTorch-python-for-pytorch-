#import warnings
#warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import os
import random
from tensorflow import keras
from tensorflow.python.ops.rnn import static_rnn
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell

vocab_size=10000
max_seq_num = 256	#句子最大长度
num_dimensions = 50 #词向量长度 
batch_size = 64 # batch的尺寸
num_labels = 2  # 输出的类别数
iterations = 1000 # 迭代的次数 
dropout_keep_prob = 0.5  # dropout保留比例
learning_rate = 1e-3

#1. download imdb data and show data examples
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
print('imdb data lens:%d,%d'%(len(train_data[0]), len(train_data[1])))
print(train_data[0])
print(train_labels[0])

# 扩展整数数组让他们拥有相同的长度，在sequence后面扩充0
train_data = keras.preprocessing.sequence.pad_sequences(train_data,maxlen=max_seq_num) 
test_data = keras.preprocessing.sequence.pad_sequences(test_data,maxlen=max_seq_num)
print(train_data[0])

#2. setup model
"""
构建tensorflow图

"""
tf.reset_default_graph()
X_holder = tf.placeholder(tf.int32, [None, max_seq_num])
Y_holder = tf.placeholder(tf.int32, [None])
embedding = tf.get_variable('embedding', [vocab_size, num_dimensions])
embedding_inputs = tf.nn.embedding_lookup(embedding, X_holder)

# lstm模型
# 0.原始输入数据格式：batch_size,max_seq_num,num_dimensions
# 1.合并数据，rnn_input是一个max_seq_num长度的数组，数组元素是(batch_size,num_dimensions)张量
rnn_input = tf.unstack(embedding_inputs, max_seq_num, axis=1)
print("rnn_input shape:%s,%s"%(len(rnn_input),rnn_input[0].shape))
lstm_cell = BasicLSTMCell(20, forget_bias=1.0)
lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)
rnn_outputs, rnn_states = static_rnn(lstm_cell, rnn_input, dtype=tf.float32)
# rnn_outputs是一个max_seq_num长度的数组，数组元素是(batch_size,20)张量
print("rnn_outputs shape:%s,%s"%(len(rnn_outputs),rnn_outputs[0].shape))
# rnn_states是一个2长度的数组，数组元素是(batch_size,20)张量
print("rnn_states shape:%s,%s"%(len(rnn_states),rnn_states[0].shape))
logits = tf.layers.dense(rnn_outputs[-1], num_labels)
predict_Y = tf.argmax(logits, axis=1)
losses = tf.nn.softmax_cross_entropy_with_logits(
		labels=tf.one_hot(Y_holder, num_labels),
		logits = logits
		)

mean_loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mean_loss)
isCorrect = tf.equal(Y_holder, tf.cast(predict_Y, dtype=tf.int32))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

# training
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

steps = np.zeros(iterations)
ACC = np.zeros_like(steps)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
		
	print ("begin training")
	# 训练
	for step in range(iterations):
		#selected_index = random.sample(list(range(len(train_data))), k=batch_size)
		selected_index = np.random.choice(len(train_data),size=batch_size)
		batch_X = train_data[selected_index]
		batch_Y = train_labels[selected_index]
		feed_dict = {
			X_holder: batch_X,
			Y_holder: batch_Y
		}
		_, mean_loss_val,accuracy_value = sess.run([optimizer, mean_loss,accuracy], feed_dict=feed_dict)
		steps[step]=step
		ACC[step]=accuracy_value
		print("rnn_outputs shape:%s,%s"%(len(rnn_outputs),rnn_outputs[0].shape))
		if step%10 == 0:
			print ("step = {}\t mean loss ={} acc ={}".format(step, mean_loss_val,accuracy_value))

		
# # plt image
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(steps,ACC,label='acc')
# ax.set_xlabel('step')
# ax.set_ylabel('acc')
# fig.suptitle('MSE')
# handles,labels = ax.get_legend_handles_labels()
# ax.legend(handles,labels=labels)
# plt.show()

import tensorflow as tf
import pickle


def load_params():
	books, authors2int, title2int, book_tags, tags = pickle.load(open('book_params.p','rb'))
	return books, authors2int, title2int, book_tags, tags

books, authors2int, title2int, book_tags, tags = load_params()
EMBED_DIM = 32

TAG_NUM = max(tags['tag_id'])  # 34251
BOOK_NUM = max(books['book_id'])  #10000
BOOK_TAG_LENGTH = 10 #how to find the max number of longest genres list of a book


def get_book_input():
	book_id = tf.placeholder(tf.int32, [None,1], name='book_id')
	book_author = tf.placeholder(tf.int32, [None,1], name='book_author')
	book_title = tf.placeholder(tf.int32, [None,1], name='book_title')
	book_tags = tf.placeholder(tf.int32, [None,BOOK_TAG_LENGTH], name='book_tag')
	dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
	return book_id, book_author, book_title, book_tags


def get_book_id_embedding(book_id):
	with tf.name_scope('book_embedding'):
		id_embed_matrix = tf.get_variable(name='id_embed_matrix', initializer=tf.random_uniform([
			BOOK_NUM, EMBED_DIM], dtype=tf.float32))
		id_embed_layer = tf.nn.embedding_lookup(
			id_embed_matrix, book_id, name='id_embed_layer')
		return id_embed_layer

def get_book_tag_embedding(tag_id):
	commbiner = 'sum'
	with tf.name_scope('tag_embedding'):
		tag_embed_matrix = tf.get_variable(name='tag_embedding', initializer=tf.truncated_normal([
			TAG_NUM, EMBED_DIM], dtype=tf.float32))
		tag_embed_layer = tf.nn.embedding_lookup(tag_embed_matrix, tag_id, name='tag_embed_layer')
		if commbiner == 'sum':
			tag_embed_layer = tf.reduce_sum(tag_embed_layer, axis=1, keep_dims=True) #argument: keep_dims=True ?
		return tag_embed_layer


def get_book_author_embedding(book_author):
	with tf.name_scope('book_embedding'):
		author_embed_matrix = tf.get_variable(name='author_embed_matrix', initializer=tf.random_uniform([
			BOOK_NUM, EMBED_DIM], dtype=tf.float32))
		author_embed_layer = tf.nn.embedding_lookup(
			author_embed_matrix, book_author, name='author_embed_layer')
		return author_embed_layer

def get_book_title_cnn_layer(book_title, dropout_keep_prob):
	with tf.name_scope('book_embedding'):
		title_embed_matrix = tf.get_variable(name='title_embed_matrix', initializer=tf.random_uniform([
			BOOK_NUM, EMBED_DIM], dtype=tf.float32))
		title_embed_layer = tf.nn.embedding_lookup(
			title_embed_matrix, book_title, name='title_embed_matrix')
		title_embed_layer = tf.expand_dims(title_embed_layer, -1)# add a dim to tensor, change shape from (?,1,32) to (?,1,32,1)

	pool_layer_list = []
	FILTER_NUM = 8
	window_sizes = [2,3,4,5]

	for window_size in window_sizes:
		with tf.name_scope('title_conv_maxpool_{}'.format(window_size)):
			filter_weight = tf.Variable(tf.truncated_normal([
				window_size, 1, 1, FILTER_NUM], stddev=0.1), name='filter_weight')
			filter_bias = tf.constant(0.1, shape=[FILTER_NUM], name='filter_bias')
			conv_layer = tf.nn.conv2d(title_embed_layer, filter_weight, [1,1,1,1],padding='SAME', name='conv_layer')
			relu_layer = tf.nn.relu(conv_layer + filter_bias, name='relu_layer')
			pool_layer = tf.nn.max_pool(relu_layer, [1,4,1,1], [1,1,1,1],padding='SAME',name='pool_layer' )
			pool_layer_list.append(pool_layer)

	
	with tf.name_scope('dropout_layer'):
		pool_layer = tf.concat(pool_layer_list, 3, name='pool_layer')
		feature_map_number = FILTER_NUM * len(window_sizes)
		pool_layer_flat = tf.reshape(pool_layer, [-1, 1, feature_map_number], name='pool_layer_flat')
		dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name='dropout_layer')
	return dropout_layer
	

def full_connection(id_embed_layer, tag_embed_layer, author_embed_layer, title_dropout_layer):
	with tf.name_scope('full_connection'):
		id_fc_layer = tf.layers.dense(
			id_embed_layer, EMBED_DIM, name='id_fc_layer', activation=tf.nn.relu)
		tag_fc_layer = tf.layers.dense(
			tag_embed_layer, EMBED_DIM, name='tag_fc_layer', activation=tf.nn.relu)
		author_fc_layer = tf.layers.dense(
			author_embed_layer, EMBED_DIM, name='author_fc_layer', activation=tf.nn.relu)

		book_feature_layer = tf.concat([
			id_fc_layer, tag_fc_layer, author_fc_layer, title_dropout_layer], 2) #[?]
		book_feature_layer = tf.contrib.layers.fully_connected(
			book_feature_layer, 200, tf.tanh)

def book_feature_nn():
	book_id, book_author, book_title, book_tags = get_book_input()
	id_embed_layer = get_book_id_embedding(book_id)
	tag_embed_layer = get_book_tag_embedding(book_tags)
	author_embed_layer = get_book_author_embedding(book_author)
	title_dropout_layer = get_book_title_cnn_layer(book_title, 0.5)
	book_feature_layer = full_connection(id_embed_layer, tag_embed_layer, author_embed_layer, title_dropout_layer)


if __name__ == '__main__':
	book_feature_nn()
	print('end')
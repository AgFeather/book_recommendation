import tensorflow as tf
import pandas as pd


def load_data():
	user_rating = pd.read_csv('./data/ratings.csv')
	user_id = user_rating['user_id']
	rating = user_rating['rating']
	book_id  =user_rating['book_id']
	return user_id, rating, book_id


user_id, rating, book_id = load_data()
USER_NUM = max(user_id) #53424
EMBED_DIMS = 32

def get_user_input():
	user_id = tf.placeholder(tf.int32, [None, 1], name='user_id')
	rating = tf.placeholder(tf.int32, [None, 1], name='rating')
	book_id = tf.placeholder(tf.int32, [None, 1], name='book_id')
	return user_id, book_id, rating

def get_user_id_embedding(user_id):
	with tf.name_scope('user_embedding'):
		user_id_embed_matrix = tf.get_variable(
			name='user_id_embed_matrix', initializer=tf.random_normal([USER_NUM, EMBED_DIMS], dtype=tf.float32))
		user_id_embed_layer = tf.nn.embedding_lookup(user_id_embed_matrix, user_id)
	return user_id_embed_layer


if __name__ == '__main__':
	user_id, book_id, rating = get_user_input()
	user_id_embed_layer = get_user_id_embedding(user_id)
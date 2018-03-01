import tensorflow as tf
import pickle

books, authors2int, title2int, book_tags, tags = load_params()
EMBED_DIM = 32
AUTHOR_NUM = max(books['authors'])
TITLE_NUM = max(books['title'])
TAG_NUM = max([tags['tag_id']])
BOOK_NUM = max([books['book_id']])

def load_params():
	books, authors2int, title2int, book_tags, tags = 
		pickle.load(open('book_params.p', 'rb'))
	return books, authors2int, title2int, book_tags, tags

def get_book_input():
	book_id = tf.placeholder(tf.int32, [None,1], name='book_id')
	book_author = tf.placeholder(tf.int32, [None,1], name='book_author')
	book_title = tf.placeholder(tf.int32, [None,1], name='book_title')

	return book_id, book_author, book_title

def get_book_features():

	
	tag_embed_matrix = tf.Variable(tf.random_uniform([
		TAG_NUM, EMBED_DIM], dtype=tf.float32, name='tag_embed_matrix'))

def get_book_id_embedding(book_id):
	with tf.name_scope('book_embedding'):
		id_embed_matrix = tf.get_variable(tf.random_uniform([
			BOOK_NUM, EMBED_DIM], dtype=tf.float32, name='id_embed_matrix'))
		id_embed_layer = tf.embedding_lookup(
			book_id_embed_matrix, book_id, name='id_embed_layer')
		return id_embed_layer

def get_book_author_embedding(book_author):
	with tf.name_scope('book_embedding'):
		author_embed_matrix = tf.Variable(tf.random_uniform([
			AUTHOR_NUM, EMBED_DIM], dtype=tf.float32), name='author_embed_matrix')
		author_embed_layer = tf.embedding_lookup(
			author_embed_matrix, book_author, name='author_embed_layer')
		return author_embed_layer

def get_book_title_cnn_layer(book_title):
	with tf.name_scope('book_embedding'):
		title_embed_matrix = tf.Variable(tf.random_uniform([
			TITLE_NUM, EMBED_DIM], dtype=tf.float32, name='title_embed_matrix'))
		title_embed_layer = tf.embedding_lookup(
			title_embed_matrix, book_title, name='title_embed_matrix')
		

	


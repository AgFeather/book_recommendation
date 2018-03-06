import tensorflow as tf
import book_feature_processing
import user_feature_processing


tf.reset_graph


def model_training():

	book_id, book_author, book_title, book_tags = get_book_input()
	id_embed_layer = get_book_id_embedding(book_id)
	tag_embed_layer = get_book_tag_embedding(book_tags)
	author_embed_layer = get_book_author_embedding(book_author)
	title_dropout_layer = get_book_title_cnn_layer(book_title, 0.5)
	book_feature_layer = full_connection(id_embed_layer, tag_embed_layer, author_embed_layer, title_dropout_layer)

	user_id, book_id, rating = get_user_input()
	user_id_embed_layer = get_user_id_embedding(user_id)
	
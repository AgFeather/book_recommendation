import os
import pandas as pd


def data_load():
	'''
	load data from csv files
	
	book_tags: goodreads_book_id, tag_id, count
	books: book_id, goodreads_book_id, best_book_id, work_id, books_count, isbn,
			isbn13, authors, original_publication_year, original_title, ratings_count,
			work_ratings_count work_text_review_count, ratings_1, ratings_2, ratings_3,
			ratings_4, ratings_5, image_url, small_image_url
	ratings: user_id, book_id, ratings
	tags: tag_id, tag_name
	to_read: user_id, book_id
	'''

	data_path = './data/'
	books_path = data_path+'books.csv'
	book_tags_path = data_path+'book_tags.csv'
	tags_path = data_path+'tags.csv'
	# book_tags = pd.read_csv(data_path + 'book_tags.csv')
	# books = pd.read_csv(data_path + 'books.csv')
	# ratings = pd.read_csv(data_path + 'ratings.csv')
	# tags = pd.read_csv(data_path + 'tags.csv')
	# to_read = pd.read_csv(data_path + 'to_read.csv')
	books = books_feature_processing(books_path, book_tags_path, tags_path)



def books_feature_processing(books_path, book_tags_path, tags_path):
	books = pd.read_csv(books_path)
	book_items = ['book_id','goodreads_book_id','authors',
		'original_title','ratings_count','ratings_1','ratings_2','ratings_3','ratings_4','ratings_5']
#	book_regex = 'book_id|goodreads_book_id|authors|'+\
#		'original_title|ratings_count|ratings_1|ratings_2|ratings_3|ratings_4|ratings_5'
#	books = books.filter(regex=book_regex)
	books = books.filter(items=book_items)

	authors2int = {val:i for i, val in enumerate(books['authors'])}
	books['authors'] = books['authors'].map(authors2int)
	
	title2int = {val:i for i, val in enumerate(books['original_title'])}
	books['original_title'] = books['original_title'].map(title2int)

	ratings = (1 * books['ratings_1'] + 2 * books['ratings_2'] + 3 * books['ratings_3'] + 4 * books['ratings_4'] + 5 * books['ratings_5'])/books['ratings_count']
	books['ratings_1'] = ratings
#	del books['ratings_1']
	del books['ratings_2']
	del books['ratings_3']
	del books['ratings_4']
	del books['ratings_5']
	del books['ratings_count']

#	books = pd.concat([books, ratings], axis=1)
#	print(books.head())


	'''
	将book表和book_tags表合并，得到带有书籍风格标签的表
	'''
	books_tags = pd.read_csv(book_tags_path)
	print(books_tags.head())



	return books



def user_feature_processing():
	pass


if __name__ == '__main__':
	data_load()
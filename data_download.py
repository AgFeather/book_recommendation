import zipfile
import os
from urllib.request import urlretrieve
from tqdm import tqdm


#下载数据集并解压
def data_download():
	url="https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/goodbooks-10k.zip"
	save_path = './goodbooks-10k.zip'
	data_name = 'goodbooks-10k.zip'
	if os.path.exists(save_path+data_name):
		print('found {} data'.format(data_name))
	else:
		with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(data_name)) as pbar:
			urlretrieve(url, save_path, pbar.hook)


def data_extract():
	data_path = './goodbooks-10k.zip'
	unzip_path = './data'
	data_name = 'goodbooks-10k.zip'
	extract_function = unzip
	if os.path.exists(unzip_path):
		os.makedirs(unzip_path)
	try:
		extract_function(data_name, data_path, unzip_path)
	except Exception as err:
		shutil.rmtree(unzip_path)
		raise err
	print('extracting done')


def unzip(data_name, from_path, to_path):
	print('extracting {} ...'.format(data_name))
	with zipfile.ZipFile(from_path) as zf:
		zf.extractall(to_path)


class DLProgress(tqdm):
	"""Handle progress bar while downloading"""
	last_block = 0
	def hook(self, block_num=1, block_size=1, total_size=None):
		"""
		a hook function that will be  called once on establishment
		of the network connection and once after each block read
		thereafter parameter:
		block_num: acount of blocks transferred so far
		block_size:block size in bytes
		total_size: total size of the file. this may be -1 on older
		FTP servers with do not return a file size in response to 
		retrieval reqeust
		"""
		self.total = total_size
		self.update((block_num - self.last_block) * block_size)
		self.last_block = block_num
		

if __name__ == '__main__':
	data_download()
	data_extract()

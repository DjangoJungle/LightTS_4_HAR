import os
import urllib.request
import zipfile

def download_and_extract_data():
    if not os.path.exists('UCI HAR Dataset'):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
        filename = 'UCI_HAR_Dataset.zip'
        print('Downloading UCI-HAR Dataset...')
        urllib.request.urlretrieve(url, filename)
        print('Extracting files...')
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(filename)
        print('Dataset downloaded and extracted.')
    else:
        print('Dataset already exists.')

if __name__ == '__main__':
    download_and_extract_data()
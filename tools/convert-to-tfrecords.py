import os
import random
import numpy as np
import tensorflow as tf


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

## 데이터 경로설정
LABEL_CSV = os.path.join(BASE_PATH, '../image-data/labels-map.csv')
LABEL_FILE = os.path.join(BASE_PATH, '../datasets/labels/512-common-hangul.txt')
OUTPUT_DIR = os.path.join(BASE_PATH, '../tfrecords-output')
NUM_SHARDS_TRAIN = 3  ## 훈련용
NUM_SHARDS_TEST = 1  ## 테스트용

'''
[ shards ]는 대규모 데이터셋을 분산 시스템에서 작은 파티션으로 분할한 것
확장성과 성능을 개선하기 위한 목적으로 사용
'''


def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))


#  TFRecord 파일은 TensorFlow에서 대규모 데이터셋을 저장하는 데 일반적으로 사용되는 형식
class TFRecordsConverter():

    def __init__(self, labels_csv, label_file, output_dir, num_shards_train, num_shards_test):

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.num_shards_train = num_shards_train
        self.num_shards_test = num_shards_test

        # Get lists of images and labels.
        self.filenames, self.labels = self.process_image_labels(labels_csv, label_file)

        # Counter for total number of images processed.
        self.counter = 0


    def process_image_labels(self, labels_csv, label_file):

        '''
        이미지와 레이블 (shuffled)리스트를 생성 
        images 리스트의 각 이미지의 인덱스는 labels 리스트의 동일한 인덱스에 해당하는 레이블을 가짐'''
        with open(label_file, 'r', encoding='utf-8') as labels_file:

            # Map characters to indices.
            label_dict = {}
            count = 0

            for label in labels_file:
                label = label.strip()
                label_dict[label] = count
                count += 1

        with open(labels_csv, 'r', encoding='utf-8') as labels_csv:
            
            # Build the lists.
            images = []
            labels = []

            for row in labels_csv:
                file, label = row.strip().split(',') 
                images.append(file)
                labels.append(label_dict[label])


        # Randomize the order of all the images/labels.
        shuffled_indices = list(range(len(images)))
        random.seed(12121)
        random.shuffle(shuffled_indices)

        filenames = [images[i] for i in shuffled_indices]
        labels = [labels[i] for i in shuffled_indices]

        return filenames, labels



    def write_tfrecords_file(self, output_path, indices):

        # Writes out TFRecords file.
        with tf.io.TFRecordWriter(output_path) as writer:

            for i in indices:
                filename = self.filenames[i]
                label = self.labels[i]
                with tf.io.gfile.GFile(filename, 'rb') as f:
                    im_data = f.read()

                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        'image/class/label': _int64_feature(label),
                        'image/encoded': _bytes_feature(im_data)
                        }))
                writer.write(example.SerializeToString())
                
                self.counter += 1
                
                if not self.counter % 1000:
                    print(f'Processed {self.counter} images...')

            writer.close()



    def convert(self):

        num_files_total = len(self.filenames)

        # Allocate about 15 percent of images to testing.
        num_files_test = int(num_files_total * 0.15)

        # About 85 percent will be for training.
        num_files_train = num_files_total - num_files_test

        print('Processing training set TFRecords...')

        # files_per_shard 변수는 num_files_train (학습에 사용되는 파일의 총 개수)를
        # self.num_shards_train (학습에 사용되는 샤드의 개수)로 나눈 후 올림하여 계산

        files_per_shard = tf.cast(tf.math.ceil(num_files_train / self.num_shards_train), dtype=tf.int32)
        start = 0

        for i in range(1, self.num_shards_train):
            shard_path = os.path.join(self.output_dir, f'train-{str(i)}.tfrecords')
            file_indices = np.arange(start, start + files_per_shard, dtype=int)
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, file_indices)

        # The remaining images will go in the final shard.
        file_indices = np.arange(start, num_files_train, dtype=int)
        final_shard_path = os.path.join(self.output_dir, f'train-{str(self.num_shards_train)}')
        self.write_tfrecords_file(final_shard_path, file_indices)

        print('Processing testing set TFRecords...')

        files_per_shard = tf.cast(tf.math.ceil(num_files_test / self.num_shards_test), dtype=tf.int32)
        start = num_files_train

        for i in range(1, self.num_shards_test):
            shard_path = os.path.join(self.output_dir, f'test-{str(i)}.tfrecords')
            file_indices = np.arange(start, start + files_per_shard, dtype=int)
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, file_indices)

        # The remaining images will go in the final shard.
        file_indices = np.arange(start, num_files_total, dtype=int)
        final_shard_path = os.path.join(self.output_dir, f'test-{str(self.num_shards_test)}.tfrecords')
        self.write_tfrecords_file(final_shard_path, file_indices)

        print(f'\nProcessed {self.counter} total images...')
        print(f'Number of training examples: {num_files_train}')
        print(f'Number of testing examples: {num_files_test}')
        print(f'TFRecords files saved to {self.output_dir}')



if __name__ == '__main__':

    converter = TFRecordsConverter(LABEL_CSV,
                                   LABEL_FILE,
                                   OUTPUT_DIR,
                                   NUM_SHARDS_TRAIN,
                                   NUM_SHARDS_TEST)
    converter.convert()

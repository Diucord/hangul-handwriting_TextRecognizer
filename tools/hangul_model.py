import io
import os
import tensorflow as tf

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


## 데이터 경로설정
LABEL_FILE = os.path.join(BASE_PATH, '../datasets/labels/512-common-hangul.txt')
TFRECORDS_DIR = os.path.join(BASE_PATH + '../tfrecords-output')
OUTPUT_DIR = os.path.join(BASE_PATH + '../saved-model')

MODEL_NAME = 'hangul_tensorflow'
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

NUM_EPOCHS = 3  ## 에폭(epoch) 수
BATCH_SIZE = 128 ## 한 번의 학습 단계에서 사용되는 미니배치의 크기

num_classes = 512  ## 주어진 라벨 파일의 항목 수


def _parse_function(example):

    features = tf.io.parse_single_example(
        example,
        features={
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
            'image/encoded': tf.io.FixedLenFeature([], tf.string)
        })
    label = features['image/class/label']
    image_encoded = features['image/encoded']

    # Decode the JPEG.
    image = tf.image.decode_jpeg(image_encoded, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, [IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    # Represent the label as a one hot vector.
    label = tf.one_hot(label, num_classes)
    return image, label


def main():

    with io.open(LABEL_FILE, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    global num_classes
    num_classes = len(labels)


    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print('Processing data...')


    #-------------------------------------

    # TFRecord 파일 경로목록 저장
    tf_record_pattern = os.path.join(TFRECORDS_DIR, 'train-*')
    train_data_files = tf.io.gfile.glob(tf_record_pattern)

    tf_record_pattern = os.path.join(TFRECORDS_DIR, 'test-*')
    test_data_files = tf.io.gfile.glob(tf_record_pattern)

    ## prefetch(1)을 호출하면 데이터셋은 항상 한 배치가 미리 준비되도록 
    ## 최선을 (=알고리즘이 한 배치로 작업하는 동안 이 데이터셋이 동시에 다음 배치를 준비)
    ## Create training dataset input pipeline.

    train_dataset = tf.data.TFRecordDataset(train_data_files) \
        .map(_parse_function) \
        .shuffle(1000) \
        .repeat(NUM_EPOCHS)\
        .batch(BATCH_SIZE)\
        .prefetch(1)

    test_dataset = tf.data.TFRecordDataset(test_data_files) \
        .map(_parse_function) \
        .batch(BATCH_SIZE) \
        .prefetch(1)


    #-------------------------------------
    # keras에서 사용되는 레이어(layer, 층)는 신경망 모델을 구성하는 주요한 요소이다.

    model = tf.keras.models.Sequential()   
    model.add(tf.keras.layers.Conv2D(filters=32, 
                                     kernel_size=(5, 5), 
                                     padding='same', 
                                     activation='relu', 
                                     input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), 
                                           padding='same'))

    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(5, 5), 
                                     padding='same', 
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2),padding='same'))

    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3, 3),
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(num_classes, 
                                    activation='softmax'))
    

    tf.config.run_functions_eagerly(True)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.build((IMAGE_WIDTH, IMAGE_HEIGHT, 1))
    model.summary()

    #-------------------------------------
    # Eager execution 모드 설정
    # Train the model.
    model.fit(train_dataset,
              epochs=NUM_EPOCHS)
    
    loss, acc = model.evaluate(test_dataset)
    model.save(os.path.join(OUTPUT_DIR, MODEL_NAME))


if __name__ == "__main__":
    main()
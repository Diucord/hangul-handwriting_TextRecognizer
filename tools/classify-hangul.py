import os
import io
import tensorflow as tf
import matplotlib.pyplot as plt
from math import pi
import random as rd

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'hangul_tensorflow'
LABEL_FILE = os.path.join(BASE_PATH,'../datasets/labels/512-common-hangul.txt')
SAMPLE_IMAGES = os.path.join(BASE_PATH,'../doc/source/images')
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

def load_model():
    model_path = os.path.join(BASE_PATH, 'saved-model', MODEL_NAME)
    model = tf.keras.models.load_model(model_path)
    return model

def load_labels():
    with io.open(LABEL_FILE, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()
    return labels

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
    image = tf.image.convert_image_dtype(image, tf.float32)

    # 이미지를 48px에 맞게 조정하기 위해 코드 수정
    image_resized = tf.image.resize_with_pad(image, 48, 48, antialias=True)
    image_resized = tf.image.convert_image_dtype(image_resized, tf.float32)
    image_resized = tf.image.adjust_contrast(image_resized, 2.0)  # 대비 조정
    
    # 64x64px 크기로 이미지 중앙에 배치
    image_padded = tf.image.pad_to_bounding_box(image_resized, 8, 8, 64, 64)
    image = tf.reshape(image_padded, [1, 64, 64, 1])  # 이미지 차원 업데이트
    return image


def classify_image(image_path, model, labels):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_label_index = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_label = labels[predicted_label_index]
    
    return predicted_label

def main():
    model = load_model()
    labels = load_labels()
    
    image_files = tf.io.gfile.glob(os.path.join(SAMPLE_IMAGES, 'hangul_*'))

    # 랜덤하게 3개의 이미지 선택
    random_images = rd.sample(image_files, 3)
        
    for image_file in random_images:
        predicted_label = classify_image(image_file, model, labels)
        print(f'이미지 확인 : {image_file}')
        print(f'AI가 예측한 글자 : {predicted_label}')


if __name__ == "__main__":
    main()
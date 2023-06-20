import glob
import io
import os
import random

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import scipy.ndimage as ndimage


BASE_PATH = os.path.dirname(os.path.abspath(__file__))


## 데이터 경로설정
LABEL_FILE = os.path.join(BASE_PATH, '../datasets/labels/512-common-hangul.txt')
FONTS_DIR = os.path.join(BASE_PATH, '../datasets/fonts')
OUTPUT_DIR = os.path.join(BASE_PATH, '../image-data')


## 한 폰트와 문자당 생성할 무작위 왜곡 이미지 수
DISTORTION_COUNT = 5


## 가로X세로 이미지크기 설정
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


def generate_hangul_images():

    # Configure Labels.
    with io.open(LABEL_FILE, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    # Get a list of the fonts.
    font_paths = glob.glob(FONTS_DIR + '/*.ttf')
    font_size = 48

    # Make Output Image Directory.
    image_dir = OUTPUT_DIR + '/hangul-images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Generate Images.
    with io.open(OUTPUT_DIR + '/labels-map.csv', 'w', encoding='utf-8') as labels_csv:
      
        total_count = 0
      
        for new_character in labels:
            for font_path in font_paths:
            
                total_count += 1

                # 이미지 생성
                image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=(0)) 
                font = ImageFont.truetype(font_path, font_size)
                drawing = ImageDraw.Draw(image)

                # 새 문자의 크기 계산
                new_w, new_h = drawing.textsize(new_character, font=font)

                # 가운데 정렬을 위한 새로운 텍스트 위치 계산
                new_text_position = ((IMAGE_WIDTH - new_w)//2, (IMAGE_HEIGHT - new_h)//2)
                vertical_adjustment = (IMAGE_HEIGHT - new_h) // 2

                # 새로운 채움 색상 설정
                new_fill_color = (255) 

                # 업데이트된 매개변수로 이미지에 새 문자 추가
                drawing.text((new_text_position[0], new_text_position[1] - vertical_adjustment),
                              new_character, fill=new_fill_color, font=font)
                file_string = f'hangul_{total_count}.jpeg'
                file_path = os.path.join(image_dir, file_string)
                
                # 이미지를 JPEG로 저장
                image.save(file_path, 'JPEG')

                # 레이블 CSV 파일에 파일 경로와 문자열 작성
                labels_csv.write(f'{file_path},{new_character}\n')


                for _ in range(DISTORTION_COUNT):

                    total_count += 1

                    file_string = f'hangul_{total_count}.jpeg'
                    file_path = os.path.join(image_dir, file_string)

                    # Generate a random NumPy array representing an image.
                    arr = np.array(image)

                    # Generate random alpha and sigma values within the defined range.
                    alpha = random.randint(30, 36)
                    sigma = random.randint(5, 6)

                    # Apply elastic distortion to the array with the generated alpha and sigma values
                    distorted_array = elastic_distort(arr, alpha=alpha, sigma=sigma)

                    # Convert the distorted array to a PIL Image.
                    distorted_image = Image.fromarray(distorted_array)
                    distorted_image.save(file_path, 'JPEG')

                    # 레이블 CSV 파일에 파일 경로와 문자열 작성
                    labels_csv.write(f'{file_path},{new_character}\n')


    print(f'Finished generating {total_count} images.')
    labels_csv.close()


def elastic_distort(image, alpha, sigma):

  ## 알파(alpha)는 변형의 강도를 조절하는 스케일링(factor) 요소
  ## 시그마(sigma) 변수는 가우시안 필터의 표준 편차

    random_state = np.random.RandomState(None)
    shape = image.shape

    dx = ndimage.gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    dy = ndimage.gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    distorted_image = ndimage.map_coordinates(image, indices, order=1).reshape(shape)

    return distorted_image


generate_hangul_images()

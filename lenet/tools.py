from PIL import Image, ImageDraw
    # PIL: Python Image Libaray, 파이썬 인터프리터에서 이미지를 처리할 수 있도록 함
    #   Image 모듈: PIL 이미지를 나타낼 수 있도록 함
    #   ImageDraw 모듈: Image형 객체에 대해 2D 그래픽 제공. 새 이미지 만들거나 annotate 하는 등 가능
import numpy as np
import matplotlib.pyplot as plt

def show_img(img_data, text):
    _img_data = img_data * 255
        # 기존에 [0, 1] 범위였던 픽셀값들을 [0, 255] 범위로 바꿈
        # (1, 1, 32, 32) 형상을 가짐

    _img_data = np.array(_img_data[0, 0], dtype=np.uint8)
        # 4차원 데이터를 2차원으로 바꿈. 첫 번째 배치 첫 번째 채널이 [32, 32] 이므로
    img_data = Image.fromarray(_img_data)
        # PIL.Image.fromarray() 함수: array 객체로부터 이미지 메모리를 생성
    draw = ImageDraw.Draw(img_data)
        # PIL.ImageDraw.Draw() 함수: 입력된 이미지에 draw를 할 수 있는 객체를 생성

    cx, cy = _img_data.shape[0] / 2, _img_data.shape[1] / 2
    if text is not None:
        draw.text((cx, cy), text)   # 이미지의 중심에 text 내용을 써넣음

    # plt.imshow(img_data)
    # plt.show()
    img_data.show()
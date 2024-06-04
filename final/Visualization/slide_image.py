import os
import sys
import time
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


def image_slide(sample_list):
    #'image' 창 생성
    cv2.namedWindow('food', cv2.WINDOW_NORMAL)
    cv2.moveWindow('food', 0,0)
    cv2.resizeWindow('food', 800, 800)
    
    cv2.namedWindow('face', cv2.WINDOW_NORMAL)
    cv2.moveWindow('face', 800, 0)
    cv2.resizeWindow('face', 800, 800)

    menu_list = ['제육', '돈까스', '국밥', '떡볶이', '파스타', '마라탕', '냉면', '칼국수', 
                 '햄버거', '피자', '김밥', '짜장면', '라면', '아구찜', '핫도그']

    # 폰트 색상 지정
    blue = (255, 0, 0)
    green= (0, 255, 0)
    red= (0, 0, 255)
    white= (255, 255, 255) 
    black= (0, 0, 0)
    # 폰트 지정
#     fontpath = "fonts/gulim.ttc"
#     font = ImageFont.truetype(fontpath, 30)
    
    webcam_captures = []
    
    # Open a video capture object
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open webcam")
        exit()
        
    s, f = cap.read()

    if s:
        cv2.imshow("face", f)
    
    for idx, img_name in enumerate(sample_list):
        img_files = os.getcwd() + "/Visualization/images/" + str(img_name) + ".jpg"
        img = cv2.imread(img_files)
        
        # 이미지에 글자 합성하기
#         img_pil = Image.fromarray(img)
#         draw = ImageDraw.Draw(img_pil)
#         draw.text((0, 0), menu_list[img_name], black)
#         img = np.array(img_pil)
    
        if img is None:
            print('Image load failed!')
            break

        cv2.imshow('food', img)
        if cv2.waitKey(3000) >= 0:
            break
        # Capture image from webcam for each image in the slideshow
        for _ in range(3):
            ret, frame = cap.read()
            if ret:
                cv2.imshow("face", frame)
                webcam_captures.append(frame)
            time.sleep(0.5)  # Sleep for 0.5 seconds

    # Release video capture object
    cap.release()
    
    cv2.destroyAllWindows()
    return webcam_captures


def final_image(idx, menu_id):
    #idx 를 실제 메뉴 index와 동치
    final_idx = menu_id[idx // 3]
    print(final_idx)
    
    #'image' 창 생성
    cv2.namedWindow('food', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('food', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
    
    menu_list = ['제육', '돈까스', '국밥', '떡볶이', '파스타', '마라탕', '냉면', '칼국수', 
                 '햄버거', '피자', '김밥', '짜장면', '라면', '아구찜', '핫도그']    
    # 폰트 색상 지정
    blue = (255, 0, 0)
    green= (0, 255, 0)
    red= (0, 0, 255)
    white= (255, 255, 255) 
    black= (0, 0, 0)
    # 폰트 지정
#     fontpath = "fonts/gulim.ttc"
#     font = ImageFont.truetype(fontpath, 30)
    
    img_files = os.getcwd() + "/Visualization/images/" + str(final_idx) + ".jpg"
    img = cv2.imread(img_files)
        
    # 이미지에 글자 합성하기
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((0, 0), "Recommended menu", black)
    img = np.array(img_pil)
    
    if img is None:
        print('Image load failed!')

    cv2.imshow('food', img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
            

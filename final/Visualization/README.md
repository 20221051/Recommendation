# Visualization
This directory includes codes for `visualization`

<img src='./visualization_pipeline.png'>

## Directory structure
```
/
    /images/
    /image_crawling.py
    /requirement.txt
    /slide_image.py
```
`images`: folder that includes virtual menu images

`image_crawling.py`: codes that collect the virtual menu images by crawling NAVER images.

`slide_image.py`: codes that show 5 recommended virtual menu image slides and capture the face for emotion analysis.

## 1. Virtual Menu Image Crawling

- `image_crawling.py` is covered with Virtual menu image crawling
- 15 menu is selected 
  ['제육', '돈까스', '국밥', '떡볶이', '파스타', '마라탕', '냉면', '칼국수', '햄버거', '피자', '김밥', '짜장면', '라면', '아구찜', '핫도그']
- Crawling virtual menu images from NAVER and save the images
- Save the crawling images -> /Visualization/images/
<img src='./virtual menu images.png'>

## 2. Virtual Menu Slide show & Face image Capture

- `slide_image.py` is covered with Virtual menu slide show and face image capture
- `image_slide`: function that shows the 5 recommended virtual menu image slides and capture face image
- `final_image`: function that shows the final consensus menu image
<img src='./Slideshow_facecapture.png'>

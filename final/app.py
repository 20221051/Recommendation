from EmotionAnalysis.emotion import *
from Personal_Recommendation.Recommend import *
from Visualization.slide_image import *
from Visualization.image_crawling import *


if __name__ == '__main__':
    
    menu_list = ['제육', '돈까스', '국밥', '떡볶이', '파스타', '마라탕', '냉면', '칼국수', 
                 '햄버거', '피자', '김밥', '짜장면', '라면', '아구찜', '핫도그']
    
    for idx, menu in enumerate(menu_list):
       print('메뉴 :', menu)
       crawl_images(menu, idx)
     
    personal = [29,'Rainy','Male',38]
    recommended_menu_id = recommend(personal)
    
    image_data_list = image_slide(recommended_menu_id)
    final_idx = analysis(image_data_list)
    final_image(final_idx, recommended_menu_id)
    

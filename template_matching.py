import os
import numpy as np
import imutils
import cv2


def template_matching():
    output = open('my_waldo_1.txt', 'w+')
    image_ids = open('datasets/ImageSets/val.txt').read().split('\n')
    print(image_ids)
    for image_id in image_ids:
        
        if image_id == '':
            continue
        
        template_paths = os.listdir('templates/waldo/')
        templates = []
        for template_path in template_paths:
            _template_path = 'templates/waldo/' + template_path + '/'
            paths = os.listdir(_template_path)
            
            for path in paths:
                path = 'templates/waldo/' + template_path + '/' + path
                template = cv2.imread(path)
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                template = cv2.Canny(template, 50, 200)
                templates.append(template)
        
        image_path = 'datasets/JPEGImages/' + image_id + '.jpg'
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for template in templates:
            height, width = template.shape[:2]
            
            if gray_image.shape[0] <= height or gray_image.shape[1] <= width:
                continue
                
            edge_image = cv2.Canny(gray_image, 50, 200)
            result = cv2.matchTemplate(edge_image, template, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
            (endX, endY) = (int((maxLoc[0] + width)), int((maxLoc[1] + height)))

            o = image_id + ' ' + str(maxVal) + ' ' + str(startX) + ' ' + str(startY) + ' ' + str(endX) + ' ' + str(endY)
            print(o)
            output.write(o + '\n')
    return
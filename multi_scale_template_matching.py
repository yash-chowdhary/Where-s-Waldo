import os
import numpy as np
import imutils
import cv2


def multi_scale_template_matching():
    output = open('my_waldo.txt', 'w+')
    image_ids = open('datasets/ImageSets/val.txt').read().split('\n')
    for image_id in image_ids:
        
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
            found = None

            for scale in np.linspace(1.5, 2.8, 20)[::-1]:
                resized_image = imutils.resize(gray_image, width=int(gray_image.shape[1] * scale))
                r = gray_image.shape[1] / float(resized_image.shape[1])

                if resized_image.shape[0] < height or resized_image.shape[1] < width:
                    break

                edge_image = cv2.Canny(resized_image, 50, 200)
                result = cv2.matchTemplate(edge_image, template, cv2.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)
                    
            if found is None:
                continue

            (maxVal, maxLoc, r) = found
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + width) * r), int((maxLoc[1] + height) * r))

            o = image_id + ' ' + str(maxVal) + ' ' + str(startX) + ' ' + str(startY) + ' ' + str(endX) + ' ' + str(endY)
            print(o)
            output.write(o + '\n')
    return
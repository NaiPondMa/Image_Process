import cv2
import numpy as np
import matplotlib.pyplot as plt


def manual_zoom_function(image, zoom_factor):
    """
    ใช้ algorithm nearest neighbor ในการ zoom ภาพ
    
    Parameters:
    image (numpy.ndarray): ภาพต้นฉบับที่ต้องการ zoom
    zoom_factor (float): อัตราการ zoom ของภาพ (ค่ามากกว่า 1 คือ zoom in, ค่าน้อยกว่า 1 คือ zoom out)

    returns:
    numpy.ndarray: ภาพที่ถูก zoom แล้ว
    """
    # step 1: ตรวจสอบว่า zoom_factor เป็นค่าที่ถูกต้อง
    if zoom_factor <= 0:
        raise ValueError("zoom_factor must be greater than 0")

    # step 2: convert ภาพเป็นขาวดํา
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # step 3: คํานวณขนาดของภาพใหม่
    new_height = int(gray_image.shape[0] * zoom_factor)
    new_width = int(gray_image.shape[1] * zoom_factor)

    # step 4: สร้างภาพใหม่ที่มีขนาดตามที่คํานวณ
    zoomed_image = np.zeros((new_height, new_width), dtype=image.dtype)

    #step 5: ใช้ algorithm nearest neighbor ในการ zoom ภาพ
    for i in range(new_height):
        for j in range(new_width):
            # คํานวณพิกัดของ pixel เริ้มต้นจากภาพที่ถูก zoom
            orig_x = int(i / zoom_factor)
            orig_y = int(j / zoom_factor)

            # ตรวจสอบว่าพิกัดต้นฉบับอยู่ในขอบเขตของภาพต้นฉบับ กรณีมีก่ารปัดเศษส่วน floating
            orig_x = min(orig_x, gray_image.shape[0] - 1)
            orig_y = min(orig_y, gray_image.shape[1] - 1)

            # นำ pixel จากภาพต้นฉบับมาใส่ในภาพที่ถูก zoom
            zoomed_image[i, j] = gray_image[orig_x, orig_y]

    return zoomed_image

"""
Testing by using matplotlib to display the original and zoomed images

"""


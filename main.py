import cv2
import imutils
import numpy as np
import pytesseract
import re
import csv
from os import path, mkdir
from datetime import datetime as dt

PATH = 'files/'
ESC = 27
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Enrique Niebles\AppData\Local\Programs\Tesseract-OCR\tesseract'


def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(float(s))


def open_file():
    if not path.isdir(PATH):
        mkdir(PATH)
        with open(f'{PATH}/placas.csv', 'a', encoding='UTF8') as f:
            file = csv.writer(f)
            file.writerow(['timestamp', 'date', 'time', 'placa'])


def close_camera():
    cv2.destroyAllWindows()


def show_results(cropped_img, img, dim, show=False):
    # Extract text from image.
    text = pytesseract.image_to_string(cropped_img, config='--psm 7').strip()

    match_plate_fmt = bool(re.fullmatch(r"(\w{3}[^\w]\d{3})", text))
    x, y, w, h = dim
    pos = (x - (w // 1000), y + (h // 10) - 15)

    if match_plate_fmt and show:
        text = re.sub(r"[^\w]", '-', text).upper()
        date = dt.now()
        datetime = date.isoformat(sep=' ', timespec='seconds').split()
        prev_plate = ''
        with open(f'{PATH}/placas.csv', 'r', encoding='UTF8') as f:
            last_placa = list(csv.DictReader(f))[-1]
            prev_hour = dt.strptime(last_placa['time'], '%H:%M:%S')
            prev_plate = last_placa['placa']
            time_diff = get_sec(str(date.time())) - get_sec(str(prev_hour.time()))
            print(date, prev_hour, time_diff)
        if time_diff > 60 and prev_plate != text:
            with open(f'{PATH}/placas.csv', 'a', newline='', encoding='UTF8') as f:
                file = csv.writer(f)
                file.writerow([date.timestamp(), datetime[0], datetime[1], text])
            print([date.timestamp(), datetime[0], datetime[1], text])
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.imshow("Placa", img)
            # Press 'Enter' key to re-capture image.
            # cv2.waitKey(0)
            # cv2.destroyWindow('Placa')


def get_plate_number():
    while True:
        ret, frame = CAMERA.read()
        image = cv2.resize(frame, dsize=(640, 480), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('Entrada', image)

        # Close camera capture.
        if cv2.waitKey(1) == ESC:
            close_camera()
            break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to GREY scale.
        gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise.
        edged = cv2.Canny(gray, 30, 200)  # Canny Edge Method to perform edge detection.

        # Find contour of the plate.
        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        screen_counter = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            # Rectangular counter
            if len(approx) == 4:
                dim = cv2.boundingRect(c)
                screen_counter = approx
                break

        if screen_counter is not None:
            # Draw a green counter on a new image of the plate.
            cv2.drawContours(image, [screen_counter], -1, (0, 255, 0), 3)
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [screen_counter], 0, 255, -1, )
            cv2.bitwise_and(image, image, mask=mask)
            x, y = np.where(mask == 255)
            top_x, top_y = np.min(x), np.min(y)
            bottom_x, bottom_y = np.max(x), np.max(y)
            cropped_img = gray[top_x:bottom_x + 1, top_y:bottom_y + 1]

            show_results(cropped_img, image, dim, show=True)


if __name__ == '__main__':
    CAMERA = cv2.VideoCapture(0)
    # Verify access to the camera.
    if CAMERA.isOpened():
        open_file()
        f = open(f'{PATH}/placas.csv', 'a', encoding='UTF8')
        file = csv.writer(f)
        get_plate_number()

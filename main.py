import cv2
import imutils
import numpy as np
import pytesseract
import re

ESC = 27
YELLOW_LB = [22, 93, 0]
YELLOW_UP = [45, 255, 255]
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Enrique Niebles\AppData\Local\Programs\Tesseract-OCR\tesseract'


def close_camera():
    print('Cerrando cámara...')
    cv2.destroyAllWindows()
    print('Cámara cerrada.')


def show_results(cropped_img, img, show=False):
    # Extract text from image.
    text = pytesseract.image_to_string(cropped_img, config='--psm 7')

    match_plate_fmt = bool(re.fullmatch(r"(\w{3}[\s\-\*\.]\d{3})", text.strip()))

    if match_plate_fmt and show:
        print("La placa detectada es:", text)
        cv2.imshow("Frame", img)
        cv2.imshow('Placa', cropped_img)
        # Press 'Enter' key to re-capture image.
        cv2.waitKey(0)
        cv2.destroyWindow('Frame')
        cv2.destroyWindow('Placa')


def get_plate_number():
    print('Obteniendo información de la placa...')
    while True:
        ret, frame = CAMERA.read()

        image = cv2.resize(frame, dsize=(640, 480), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        original = image.copy()
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
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # Rectangular counter
            if len(approx) == 4:
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

            show_results(cropped_img, image, show=True)

        print("No hay contorno detectado.")


if __name__ == '__main__':
    CAMERA = cv2.VideoCapture(0)
    # Verify access to the camera.
    if CAMERA.isOpened():
        get_plate_number()

    print("No se pudo abrir la cámara. Verifique los permisos.")

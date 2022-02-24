import cv2
import imutils
import numpy as np
import pytesseract

ESC = 27


def close_camera():
    print('Cerrando cámara...')
    cv2.destroyAllWindows()
    print('Cámara cerrada.')


def get_plate_number():
    print('Obteniendo información de la placa...')
    while True:
        ret, frame = CAMERA.read()

        image = cv2.resize(frame, dsize=(640, 480), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('Input', image)

        key_pressed = cv2.waitKey(1)
        save_key = key_pressed & 0xFF

        # Close camera capture.
        if key_pressed == ESC:
            close_camera()
            break

        # Press 's' key to save and analyze image.
        if save_key == ord('s'):
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
                approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                # Rectangular counter
                if len(approx) == 4:
                    screen_counter = approx
                    break

            if screen_counter is None:
                print("No hay contorno detectado.")
                continue

            # Draw a green counter on a new image of the plate.
            cv2.drawContours(image, [screen_counter], -1, (0, 255, 0), 3)
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screen_counter], 0, 255, -1, )
            new_image = cv2.bitwise_and(image, image, mask=mask)
            x, y = np.where(mask == 255)
            top_x, top_y = np.min(x), np.min(y)
            bottom_x, bottom_y = np.max(x), np.max(y)
            cropped_img = gray[top_x:bottom_x + 1, top_y:bottom_y + 1]

            # Extract text from image.
            pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Enrique Niebles\AppData\Local\Programs\Tesseract-OCR\tesseract'
            text = pytesseract.image_to_string(cropped_img, config='--psm 11')[
                   0:7]  # Colombian plate has only 7 characters.
            print("La placa detectada es:", text)
            cv2.imshow("Frame", image)
            cv2.imshow('Placa', cropped_img)

            # Press 'Enter' key to re-capture image.
            cv2.waitKey(0)
            cv2.destroyWindow('Frame')
            cv2.destroyWindow('Placa')


if __name__ == '__main__':
    CAMERA = cv2.VideoCapture(0)
    # Verify access to the camera.
    if CAMERA.isOpened():
        get_plate_number()

    print("No se pudo abrir la cámara. Verifique los permisos.")

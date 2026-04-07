import cv2
import numpy as np

image = cv2.imread('minion.jpg')
if image is None:
    print("Ошибка: файл minion.jpg не найден")
    exit()

image = cv2.resize(image, dsize=(940, 580))


def add_noise(img, noise_type='gaussian', noise_factor=0.1):
    if noise_type == 'gaussian':
        row, col, ch = img.shape
        mean = 0
        sigma = noise_factor * 255
        gauss = np.random.normal(mean, sigma, size=(row, col, ch))
        noisy = img + gauss
        return np.clip(noisy, a_min=0, a_max=255).astype(np.uint8)
    return img


noisy_image = add_noise(image, noise_type='gaussian', noise_factor=0.1)
cv2.imshow('Noisy Image', cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY))

denoised_image = cv2.medianBlur(noisy_image, 5)
cv2.imshow('Denoised Image', cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY))

cropped = denoised_image[100:500, 200:600]
cv2.imshow('Cropped Image', cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY))

gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((5, 5), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

final_result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

if contours:
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        print(f"Площадь объекта: {area}")
        print(f"Центр объекта: x={cx}, y={cy}")

        cv2.drawContours(final_result, [cnt], -1, (0, 255, 0), 3)
        cv2.circle(final_result, (cx, cy), 7, (0, 0, 255), -1)

cv2.imshow('Final Result (B&W Background)', final_result)
cv2.imshow('Threshold (Morph)', morph)

cv2.waitKey(0)
cv2.destroyAllWindows()

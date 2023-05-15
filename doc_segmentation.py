import cv2


PATH = "doc.jpg"
blur_kernel_size = 3

# загрузка изображения, конвертация в оттенки серого и бинаризация
image = cv2.imread(PATH)
image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binar = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# удаление шумов
filtered = cv2.medianBlur(binar, blur_kernel_size)

# задание размера ядра для объединения связных областей
kernel_size = (10, 7)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
close = cv2.dilate(filtered, kernel, iterations=1)

# выделение контуров
contours, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# отрисовка контуров
for c in contours:
    # Получение координат прямоугольника, описывающего контур
    x, y, w, h = cv2.boundingRect(c)

    # Рисование прямоугольника на изображении
    cv2.rectangle(image_rgba, (x, y), (x + w, y + h), (50, 0, 110, 255), -1)

# наложение с изображения с сегментацией на оригинал
out = cv2.addWeighted(image, 0.5, image_rgba[:, :, :3], 0.5, 0)

# запись результата в файл
cv2.imwrite('out.png', out)

# вывод результата в окно
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800, 600)
cv2.imshow('image', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
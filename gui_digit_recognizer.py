from keras.models import load_model
from tkinter import *
import tkinter as tk
import numpy as np
import win32gui
import PIL
from PIL import ImageGrab, Image
import PIL.ImageOps

model = load_model('mnist.h5')
model_1 = load_model('keras_mnist_1.h5')


def predict_digit(img):
    print(type(img))
    img = PIL.ImageOps.invert(img)
    # изменение рзмера изобржений на 28x28
    img = img.resize((28, 28))
    # конвертируем rgb в grayscale
    img = img.convert('L')
    img = np.array(img)
    # изменение размерности для поддержки модели ввода и нормализации
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    # предстказание цифры
    res = model.predict([img])[0]
    for i in img[0]:
        for k, j in enumerate(i):
            if(k <  3):
                j[0] = 0
            if round(j[0], 1) == 0:
                print('   ', end=' ')
            else:
                print(round(j[0], 1), end=' ')
        print()
    print   (  [round(i,2) for i in res]    )
    return np.argmax(res), max(res)



def predict_digit_1(img):
    print(type(img))
    img = PIL.ImageOps.invert(img)
    # изменение рзмера изобржений на 28x28
    img = img.resize((28, 28))
    # конвертируем rgb в grayscale
    img = img.convert('L')
    img = np.array(img)
    # изменение размерности для поддержки модели ввода и нормализации
    img = img.reshape(1, 784, 1)
    img = img / 255.0
    # предстказание цифры
    res = model_1.predict([img])[0]
    for j, i in enumerate(img[0]):
        if round(i[0], 1) == 0:
            print('   ', end=' ')
        else:
            print(round(i[0], 1), end=' ')
        if(j%28 == 0):
            print()
    print()
    print   (  [round(i,2) for i in res]    )
    return np.argmax(res), max(res)



class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Создание элементов
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Думаю..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Распознать", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Очистить", command=self.clear_all)

        # Сетка окна
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        #self.canvas.bind("", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        '''
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)  # получаем координату холста
        im = ImageGrab.grab(rect)
        '''
        x, y = (self.canvas.winfo_rootx(), self.canvas.winfo_rooty())
        width, height = (self.canvas.winfo_width(),
                         self.canvas.winfo_height())
        a, b, c, d = (x+15, y+15, x + width+15, y + height+15)
        im = ImageGrab.grab(bbox=(a, b, c, d))


        digit, acc = predict_digit_1(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


app = App()
mainloop()
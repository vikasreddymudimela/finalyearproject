import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk

window = tk.Tk()

window.title("Fruit Adulteration Detection")

window.geometry("500x700")#500x510
window.configure(background ="gray51")

title = tk.Label(text="Click below to choose picture for testing Adulteration....", background = "lightgreen", fg="Brown", font=("", 17))
title.grid()
def BlackRotCankerApple():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Fruit Adulteration Detection")

    window1.geometry("500x510")
    window1.configure(background="gray51")

    def exit():
        window1.destroy()
    rem = "The remedies for BlackRotCanker Spot are:\n\n "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " remove mummified fruit and sanitize with Thiophanate-Methyl spray"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                        fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()

def BrownRotApple():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Fruit Adulteration Detection")

    window1.geometry("650x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for BrownRot are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Thiophanate-Methyl spray,Organic Spray,Captan sanitizing spray"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()
def ScabApple():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Fruit Adulteration Detection")

    window1.geometry("650x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for scab are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = "cooper- and sulphur- based fungicides"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()

def AnthracNoseBanana():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Fruit Adulteration Detection")

    window1.geometry("650x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for AnthracNose are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Monitor the field, handpick adulterated\n  Demethyldehydropodophyllotoxin or picropodophyllone"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()

def BlackRottedBanana():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Fruit Adulteration Detection")

    window1.geometry("520x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for BlackRotted are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)

    rem1 = " Monitor the field, remove and destroy infected leaves. \n  carbendazim 50% wp , chlorothalonil 75%  ,  propiconazole 25%(choose and apply one crop)"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()
def SpeckleBanana():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Fruit Adulteration Detection")

    window1.geometry("500x510")
    window1.configure(background="gray51")

    def exit():
        window1.destroy()
    rem = "The remedies for Speckle  are:\n\n "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Frenta 750 WG ,Manzate DF,Penncozeb ,Penncozeb 750 DF ,Rotam Winner Mancozeb WP,Unizeb Disperss 750 DF,Mancoflo 420 SC ,Arysta LifeScience Mancozeb 750 WG"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                        fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()

def FungalOranges():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Fruit Adulteration Detection")

    window1.geometry("650x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for Fungal are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Horticultural Oils,dormant spray"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()
def MelanoseOranges():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Fruit Adulteration Detection")

    window1.geometry("650x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for Melanose are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Searles Copper Oxychloride"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()

def PencililliumDigitatumOranges():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Fruit Adulteration Detection")

    window1.geometry("650x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for PencililliumDigitatum are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = "Orchard and packinghouse sanitation is required"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()

def PencililliumMoldOranges():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Fruit Adulteration Detection")

    window1.geometry("520x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for PencililliumMold  are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)

    rem1 = " postharvest PYR treatment"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


##########################################################################################
def AnthracNosePapaya():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Fruit Adulteration Detection")

    window1.geometry("520x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for AnthracNosePapaya  are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)

    rem1 = " postharvest PYR treatment"             #### add treatment
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


def PowderyMildewPapaya():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Fruit Adulteration Detection")

    window1.geometry("520x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for PowderyMildewPapaya  are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)

    rem1 = " postharvest PYR treatment"                  ##add treatment
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()



def FungalWatermelon():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Fruit Adulteration Detection")

    window1.geometry("520x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for FungalWatermelon  are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)

    rem1 = " postharvest PYR treatment"                  ##add treatment
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()




def AnthracNoseWatermelon():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Fruit Adulteration Detection")

    window1.geometry("520x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for AnthracNoseWatermelon  are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)

    rem1 = " postharvest PYR treatment"                 ##add treatment
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()
    

def analysis():
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
    verify_dir = 'testpicture'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'FruitDisease-{}-{}.model'.format(LR, '2conv-basic')

    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
##    tf.reset_default_graph()
    tf.compat.v1.reset_default_graph()
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 19, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        model_out = model.predict([data])[0]
        print(model_out)
        print('model {}'.format(np.argmax(model_out)))

        if np.argmax(model_out) == 0:
            str_label = 'BlackRotCankerApple'
        elif np.argmax(model_out) == 1:
            str_label = 'BrownRotApple'
        elif np.argmax(model_out) == 2:
            str_label = 'ScabApple'
        elif np.argmax(model_out) == 3:
            str_label = 'AnthracNoseBanana'
        elif np.argmax(model_out) == 4:
            str_label = 'BlackRottedBanana'
        elif np.argmax(model_out) == 5:
            str_label = 'SpeckleBanana'

        elif np.argmax(model_out) == 6:
            str_label = 'FungalOranges'
        elif np.argmax(model_out) == 7:
            str_label = 'MelanoseOranges'
        elif np.argmax(model_out) == 8:
            str_label = 'PencililliumDigitatumOranges'
        elif np.argmax(model_out) == 9:
            str_label = 'PencililliumMoldOranges'
        elif np.argmax(model_out) == 10:
            str_label = 'FreshApples'
        elif np.argmax(model_out) == 11:
            str_label = 'FreshOranges'

        elif np.argmax(model_out) == 12:
            str_label = 'FreshBanana'
            
        elif np.argmax(model_out) == 13:
            str_label = 'AnthracNosePapaya'
        elif np.argmax(model_out) == 14:
            str_label = 'Fresh_Papaya'
        elif np.argmax(model_out) == 15:
            str_label = 'PowderyMildewPapaya'
        elif np.argmax(model_out) == 16:
            str_label = 'FungalWatermelon'
        elif np.argmax(model_out) == 17:
            str_label = 'FreshWatermelon'
        elif np.argmax(model_out) == 18:
            str_label = 'AnthracNoseWatermelon'
##        message = tk.Label(text='Status: '+status, background="lightgreen",
##                           fg="Brown", font=("", 15))
##        message.grid(column=0, row=3, padx=10, pady=10)
        status=''
        if str_label == 'BlackRotCankerApple':
            Adulterationname = "BlackRotCankerApple  "
            Adulteration = tk.Label(text='Fruit name : ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='BlackRotCankerApple', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=BlackRotCankerApple)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)
        elif str_label == 'BrownRotApple':
            Adulterationname = "BrownRotApple"
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='BrownRotApple', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=BrownRotApple)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)
        elif str_label == 'ScabApple':
            Adulterationname = "ScabApple "
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='ScabApple', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=ScabApple)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)



        elif str_label == 'AnthracNoseBanana':
            Adulterationname = "AnthracNoseBanana  "
            Adulteration = tk.Label(text='Fruit name : ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='AnthracNoseBanana', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=AnthracNoseBanana)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)
        elif str_label == 'BlackRottedBanana':
            Adulterationname = "BlackRottedBanana"
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='BlackRottedBanana', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=BlackRottedBanana)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)
        elif str_label == 'SpeckleBanana':
            Adulterationname = "SpeckleBanana "
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='SpeckleBanana', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=SpeckleBanana)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)



#######
        elif str_label == 'FungalOranges':
            Adulterationname = "FungalOranges  "
            Adulteration = tk.Label(text='Fruit name : ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='FungalOranges', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=FungalOranges)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)
        elif str_label == 'MelanoseOranges':
            Adulterationname = "MelanoseOranges"
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='MelanoseOranges', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=MelanoseOranges)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)
        elif str_label == 'PencililliumDigitatumOranges':
            Adulterationname = "PencililliumDigitatumOranges "
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='PencililliumDigitatumOranges', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=PencililliumDigitatumOranges)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)

        elif str_label == 'PencililliumMoldOranges':
            Adulterationname = "PencililliumMoldOranges "
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='PencililliumMoldOranges', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=PencililliumMoldOranges)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)
######################################################################################################################################

        elif str_label == 'AnthracNosePapaya':
            Adulterationname = "AnthracNosePapaya "
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='AnthracNosePapaya', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=AnthracNosePapaya)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)

        elif str_label == 'AnthracNoseWatermelon':
            Adulterationname = "AnthracNoseWatermelon "
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='AnthracNoseWatermelon', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=AnthracNoseWatermelon)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)
        elif str_label == 'PowderyMildewPapaya':
            Adulterationname = "PowderyMildewPapaya "
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='PowderyMildewPapaya', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=PowderyMildewPapaya)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)
        elif str_label == 'FungalWatermelon':
            Adulterationname = "FungalWatermelon "
            Adulteration = tk.Label(text='Fruit Name: ' + Adulterationname, background="lightgreen",
                               fg="Black", font=("", 15))
            Adulteration.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='FungalWatermelon', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=FungalWatermelon)
            button3.grid(column=0, row=6, padx=10, pady=10)
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=9, padx=20, pady=20)
        



        elif str_label == 'FreshApples' or str_label == 'FreshBanana' or str_label =='FreshOranges' or str_label =='FreshWatermelon' or str_label =='Fresh_Papaya':
            status= 'Healthy' + str_label
            message = tk.Label(text='Status: '+status, background="gray51",
                           fg="Brown", font=("", 15))
            message.grid(column=1, row=3, padx=15, pady=15)
            r = tk.Label(text='Fruit is healthy', background="lightgreen", fg="Black",font=("", 15))
            r.grid(column=0, row=4, padx=10, pady=10)
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=9, padx=20, pady=20)


        message = tk.Label(text='Status: '+status, background="lightgreen",
                           fg="Brown", font=("", 15))
        message.grid(column=0, row=3, padx=10, pady=10)

def openphoto():
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
    # C:/Users/sagpa/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
    fileName = askopenfilename(initialdir='F:\\fruitdisease_5_fruit\\fruitdisease_5_fruit\\dataset\\test', title='Select image for analysis ',
                           filetypes=[('image files', '.png')])
    dst = "F:\\fruitdisease_5_fruit\\fruitdisease_5_fruit\\testpicture"
    print(fileName)
    print (os.path.split(fileName)[-1])
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="400", width="500")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady = 10)
    title.destroy()
    button1.destroy()
    button2 = tk.Button(text="Analyse Image", command=analysis)
    button2.grid(column=0, row=2, padx=10, pady = 10)
button1 = tk.Button(text="Get Photo", command = openphoto)
button1.grid(column=0, row=1, padx=10, pady = 10)



window.mainloop()




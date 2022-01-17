import numpy as np
import os
import tkinter as tk
import tkinter.filedialog as FileDialog
from tkinter import *
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


data_dir = "./pokemon/data"
test_dir = "./pokemon/test"
batch_size = 44
img_height = 32
img_width = 32
epochs = 20

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	data_dir,
	validation_split = 0.2,
	subset="training",
	seed = 123,
	image_size=(img_height, img_width),
	batch_size = batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names)

#Caching images between epochs
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

model = Sequential([
	data_augmentation,
  	layers.Conv2D(16, 3, padding='same', activation='relu'),
  	layers.MaxPooling2D(),
	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(64, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Dropout(0.2),
	layers.Flatten(),
	layers.Dense(128, activation='relu'),
	layers.Dense(num_classes)
])


model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

def predictImage(path):
	img = keras.preprocessing.image.load_img(path, target_size=(img_height, img_width))
	img_array = keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch
	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])
	print(
    	"Image most likely depicts a {} (with a {:.2f} percent confidence.)"
    	.format(class_names[np.argmax(score)], 100 * np.max(score))
	)
	ind = np.argmax(score)
	img = PhotoImage(file=path)
	imgCanvas.itemconfig(img_id, image=img)
	nameLabel.config(text="Name:	" + pokemonData[ind][0])
	typeLabel.config(text="Type:	" + pokemonData[ind][1])
	abilityLabel.config(text="Abilities	" + pokemonData[ind][2])
	hpLabel.config(text="HP 	" + str(pokemonData[ind][3]))
	atkLabel.config(text="Atk 	" + str(pokemonData[ind][4]))
	defLabel.config(text="Def 	" + str(pokemonData[ind][5]))
	spAtkLabel.config(text="Sp.Atk 	" + str(pokemonData[ind][6]))
	spDefLabel.config(text="Sp.Def 	" + str(pokemonData[ind][7]))
	speedLabel.config(text="Speed 	" + str(pokemonData[ind][8]))

def loadCallback():
	path = FileDialog.askopenfilename(parent=window, initialdir=os.getcwd(), title="Select your image", filetypes = (('png files', '*.png'),('jpeg files', '*.jpg'),('all files', '*.*')))
	if (path != () and path != ""):
		print("Selected: " + path);
		predictImage(path)

def exitCallback():
	exit()

pokemonData = []
pokemonData.append(["Bulbasaur", "Grass | Poison", "Overgrow | Chlorophyll(*)", 0, 0, 0, 1, 0, 0])
pokemonData.append(["Charmander", "Fire", "Blaze | SolarPower(*)", 0, 0, 0, 0, 0, 1])
pokemonData.append(["Pikachu", "Electric", "Static | Lightning Rod(*)", 0, 0, 0, 0, 0, 2])
pokemonData.append(["Squirtle", "Water", "Torrent | Rain Dish(*)", 0, 0, 1, 0, 0, 0])
pokeData = ["", "", "", "", "", "", "", "", ""]
path = ""

window = tk.Tk()
window.title("PokeDex")
C = Canvas(window, bg="blue", height=611, width=695)
filename = PhotoImage(file = 'bg.png')
background_label = Label(window, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
C.pack()

btn_color = '#00365C'
hover_color = '#005493'
loadBtn = tk.Button(window, text="Load image", command = loadCallback, bg=btn_color, activebackground = hover_color)
loadBtn.pack()
loadBtn.place(x = 408, y = 502, height = 48, width =108)

exitBtn = tk.Button(window, text="Exit", command = exitCallback, bg=btn_color, activebackground = hover_color)
exitBtn.pack()
exitBtn.place(x = 542, y = 502, height = 48, width =108)

imgCanvas = Canvas(window, width = 32, height = 32)
img = PhotoImage(file="./panel.png")
img_id = imgCanvas.create_image(0,0,anchor=NW, image=img)
imgCanvas.place(x=410, y=150)

nameLabel = tk.Label(window, textvariable = pokeData[0], bg = 'black', fg='white')
nameLabel.place(x = 64, y = 120)
typeLabel = tk.Label(window, textvariable = pokeData[1], bg = 'black', fg='white')
typeLabel.place(x = 64, y = 140)
abilityLabel = tk.Label(window, textvariable = pokeData[2], bg = 'black', fg='white')
abilityLabel.place(x = 64, y = 160)
hpLabel = tk.Label(window, textvariable = pokeData[3], bg = 'black', fg='white')
hpLabel.place(x = 64, y = 180)
atkLabel = tk.Label(window, textvariable = pokeData[4], bg = 'black', fg='white')
atkLabel.place(x = 64, y = 200)
defLabel = tk.Label(window, textvariable = pokeData[5], bg = 'black', fg='white')
defLabel.place(x=64, y = 220)
spAtkLabel = tk.Label(window, textvariable = pokeData[6], bg = 'black', fg='white')
spAtkLabel.place(x=64, y=240)
spDefLabel = tk.Label(window, textvariable = pokeData[7], bg = 'black', fg='white')
spDefLabel.place(x=64, y=260)
speedLabel = tk.Label(window, textvariable = pokeData[8], bg = 'black', fg='white')
speedLabel.place(x=64, y=260)
window.mainloop()
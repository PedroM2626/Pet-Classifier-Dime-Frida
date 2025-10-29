# %% [markdown]
# # Configuração Inicial e Importações
# Configura o logger para depuração e importa as bibliotecas necessárias.

# %%
import logging
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # Usar o backend 'Agg' para evitar problemas de exibição e salvamento
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import keras
from matplotlib.pyplot import imshow
from tensorflow.keras.layers import Dense, Flatten

# Configurar o logger
logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Script dime_frida_classifier.py iniciado.")

# %% [markdown]
# # Funções Auxiliares
# Define funções para carregar imagens e gerar mapas de calor Grad-CAM.

# %%
# helper function to load image and return it and input vector
def get_image(path):
    try:
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        return img, x
    except Exception as e:
        logging.warning(f"Aviso: Não foi possível carregar a imagem {path}. Erro: {e}")
        return None, None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last convolutional layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last convolutional layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen) with respect
    # to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array by "how important this channel is"
    # with respect to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# %% [markdown]
# # Configurações do Dataset
# Define o caminho raiz do dataset e as categorias de imagens.

# %%
# --- Configurações do Dataset ---
root = r'c:\Users\pedro\Downloads\my_precious-dataset\my_precious-dataset'
logging.info(f"Caminho root configurado: {root}")
train_split = 0.8 # 80% para treinamento
val_split = 0.2   # 20% para validação

# Definir as categorias explicitamente
categories = [os.path.join(root, 'Dime'), os.path.join(root, 'Frida')]
category_names = ['Dime', 'Frida'] # Nomes das classes para exibição

logging.info(f"Categorias encontradas: {category_names}")

# %% [markdown]
# # Carregamento e Pré-processamento dos Dados
# Carrega as imagens, as pré-processa e divide em conjuntos de treinamento, validação e teste.

# %%
# --- Carregamento e Pré-processamento dos Dados ---
data = []
logging.info(f"Iniciando carregamento de imagens para as categorias: {category_names}")
for c_idx, category_path in enumerate(categories):
    images = [os.path.join(dp, f) for dp, dn, filenames
              in os.walk(category_path) for f in filenames
              if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    logging.info(f"Número de imagens encontradas em {category_path}: {len(images)}")
    for img_path in images:
        img, x = get_image(img_path)
        if img is not None and x is not None:
              data.append({'x':x, 'y':c_idx})

logging.info(f"Comprimento final da lista de dados após o carregamento: {len(data)}")

# Contar o número de classes
num_classes = len(category_names)

# Randomizar a ordem dos dados
random.shuffle(data)

# Dividir os dados em conjuntos de treinamento, validação e teste (70%, 15%, 15%)
random.shuffle(data)
train_split_ratio = 0.7
val_split_ratio = 0.15

idx_train = int(len(data) * train_split_ratio)
idx_val = int(len(data) * (train_split_ratio + val_split_ratio))

train = data[:idx_train]
val = data[idx_train:idx_val]
test = data[idx_val:]

# Separar dados para rótulos
x_train_list = []
y_train_list = []
for t in train:
    x_train_list.append(t["x"])
    y_train_list.append(t["y"])
x_train = np.array(x_train_list)
y_train = np.array(y_train_list)

x_val_list = []
y_val_list = []
for t in val:
    x_val_list.append(t["x"])
    y_val_list.append(t["y"])
x_val = np.array(x_val_list)
y_val = np.array(y_val_list)

x_test_list = []
y_test_list = []
for t in test:
    x_test_list.append(t["x"])
    y_test_list.append(t["y"])
x_test = np.array(x_test_list)
y_test = np.array(y_test_list)

# Normalizar dados
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Converter rótulos para vetores one-hot
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Resumo
print("Carregadas %d imagens de %d categorias"%(len(data), num_classes))
logging.info(f"Número de amostras de treinamento: {len(train)}")
logging.info(f"Número de amostras de validação: {len(val)}")
logging.info(f"Número de amostras de teste: {len(test)}")

# %% [markdown]
# # Construção e Treinamento do Modelo (VGG16 com Transfer Learning)
# Configura e treina um modelo VGG16 para classificação de imagens usando transfer learning.

# %%
# --- Construção e Treinamento do Modelo (VGG16 com Transfer Learning) ---
print("\n--- Usando VGG16 para Transfer Learning ---")
vgg = keras.applications.VGG16(weights='imagenet', include_top=True)

inp = vgg.input
new_classification_layer = Dense(num_classes, activation='softmax')
out = new_classification_layer(vgg.layers[-2].output)
model_new = Model(inp, out)

# Congelar todas as camadas, exceto a última
for l, layer in enumerate(model_new.layers[:-1]):
    layer.trainable = False
for l, layer in enumerate(model_new.layers[-1:]):
    layer.trainable = True

model_new.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_new.summary()

logging.info(f"Shape de x_train: {x_train.shape}")
logging.info(f"Shape de y_train: {y_train.shape}")
logging.info(f"Shape de x_val: {x_val.shape}")
logging.info(f"Shape de y_val: {y_val.shape}")

history2 = model_new.fit(x_train, y_train,
                         batch_size=128,
                         epochs=10,
                         validation_data=(x_val, y_val))

# %% [markdown]
# # Avaliação do Modelo
# Avalia o desempenho do modelo treinado no conjunto de testes.

# %%
# --- Avaliação do Modelo ---
loss, accuracy = model_new.evaluate(x_test, y_test, verbose=0)
print('Perda no teste:', loss)
print('Precisão no teste:', accuracy)

# %% [markdown]
# # Análise de Imagem da Pasta 'tests'
# Processa imagens da pasta 'tests', faz previsões e salva os resultados.

# %%
# --- Análise de Imagem da Pasta 'tests' ---
logging.info("--- Análise de Imagens da Pasta 'tests' ---")
test_images_dir = os.path.join(root, 'tests')
output_dir = os.path.join(root, 'results')

# Certificar-se de que o diretório de saída existe
os.makedirs(output_dir, exist_ok=True)

image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    logging.warning(f"Nenhuma imagem encontrada na pasta de testes: {test_images_dir}")
    print(f"Nenhuma imagem encontrada na pasta de testes: {test_images_dir}")
else:
    for image_filename in image_files:
        img_path = os.path.join(test_images_dir, image_filename)
        logging.debug(f"Processando imagem de teste: {img_path}")
        img, img_processed = get_image(img_path)

        if img is not None and img_processed is not None:
            img_processed = np.expand_dims(img_processed, axis=0)
            prediction = model_new.predict(img_processed)
            predicted_class_idx = np.argmax(prediction)
            predicted_class_name = category_names[predicted_class_idx]

            # Salvar as imagens
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))

            ax.imshow(img)
            ax.set_title(f"Predicted: {predicted_class_name}")
            ax.axis("off")

            output_filename = os.path.splitext(image_filename)[0] + ".png"
            full_output_path = os.path.join(output_dir, output_filename)
            try:
                plt.savefig(full_output_path)
            except Exception as e:
                logging.error(f"Erro ao salvar a imagem {image_filename} em {full_output_path}: {e}")
            plt.close(fig) # Fechar a figura para liberar memória
        else:
            print(f"Não foi possível carregar ou processar a imagem de teste: {img_path}")
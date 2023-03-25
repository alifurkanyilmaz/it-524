import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

EGITIM_KLASORU = 'train'
TEST_KLASORU = 'test'
RESIM_BOYUTU = 50
OGRENME_ORANI = 1e-3
MODEL_ADI = 'kedi-kopek-ayirici'

def etiket_olustur(resim_adi):
    obje_turu = resim_adi.split('.')[-3]
    if obje_turu == 'cat':
        return np.array([1, 0])
    elif obje_turu == 'dog':
        return np.array([0, 1])
def egitim_verisi_olustur():
    olusturulan_egitim_verisi = []
    for img in tqdm(os.listdir(EGITIM_KLASORU)):
        dosya_yolu = os.path.join(EGITIM_KLASORU, img)
        resim_verisi = cv2.imread(dosya_yolu, cv2.IMREAD_GRAYSCALE)
        resim_verisi = cv2.resize(resim_verisi, (RESIM_BOYUTU, RESIM_BOYUTU))
        olusturulan_egitim_verisi.append([np.array(resim_verisi), etiket_olustur(img)])
    shuffle(olusturulan_egitim_verisi)
    np.save('egitim_verisi.npy', olusturulan_egitim_verisi)
    return olusturulan_egitim_verisi

def test_verisi_olustur():
    olusturulan_test_verisi = []
    for img in tqdm(os.listdir(TEST_KLASORU)):
        dosya_yolu = os.path.join(TEST_KLASORU, img)
        resim_no = img.split('.')[0]
        resim_verisi = cv2.imread(dosya_yolu, cv2.IMREAD_GRAYSCALE)
        resim_verisi = cv2.resize(resim_verisi, (RESIM_BOYUTU, RESIM_BOYUTU))
        olusturulan_test_verisi.append([np.array(resim_verisi), resim_no])
    shuffle(olusturulan_test_verisi)
    np.save('test_verisi.npy', olusturulan_test_verisi)
    return olusturulan_test_verisi

egitim_verisi = egitim_verisi_olustur()
test_verisi = test_verisi_olustur()

egitim = egitim_verisi[:-500]
test = egitim_verisi[-500:]

X_egitim = np.array([i[0] for i in egitim]).reshape(-1, RESIM_BOYUTU, RESIM_BOYUTU, 1)
y_egitim = [i[1] for i in egitim]
X_test = np.array([i[0] for i in test]).reshape(-1, RESIM_BOYUTU, RESIM_BOYUTU, 1)
y_test = [i[1] for i in test]


tf.compat.v1.reset_default_graph()

convnet = input_data(shape=[None, RESIM_BOYUTU, RESIM_BOYUTU, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')

convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')

convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')

convnet = regression(convnet, optimizer='adam', learning_rate=OGRENME_ORANI, loss='categorical_crossentropy',
                     name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

model.fit({'input': X_egitim}, {'targets': y_egitim}, n_epoch=10,
          validation_set=({'input': X_test}, {'targets': y_test}),
          snapshot_step=500, show_metric=True, run_id=MODEL_ADI)

fig = plt.figure(figsize=(16, 12))

for no, veri in enumerate(test_verisi[:16]):

    resim_no = veri[1]
    resim_verisi = veri[0]

    y = fig.add_subplot(4, 4, no + 1)
    orig = resim_verisi
    veri = resim_verisi.reshape(RESIM_BOYUTU, RESIM_BOYUTU, 1)
    ag_cikisi = model.predict([veri])[0]

    if np.argmax(ag_cikisi) == 1:
        str_label = 'Köpek'
    else:
        str_label = 'Kedi'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

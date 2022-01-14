"""
accuarcy
default : 0.3~0.4(accuarcy)
(28, 28, 3), trainset:100, epochs=5

mk1: 0.5~0.6
(224, 224, 3), trainset:5627, epochs=5

mk2: 0.6-0.7, 0.5-0.6, 0.7-0.8
(28, 28, 3), trainset:5627, epochs=5

mk3: 0.5~0.6
(28, 28, 1), trainset:5627, epochs=5

mk4: 0.5~0.6
(28, 28, 3), trainset:5627, epochs=10

mk5: 0.6~0.7
(28, 28, 3), trainset:5627, epochs=10, Dense layer(512) 1개 삭제

mk6: 0.6~0.7
(56, 56, 3), trainset:5627, epochs=6
"""
import os
import sys
import platform
import random

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from tensorflow import keras

img_size = (56, 56, 3)
check = True
img_save = True

def load_path(img_dir):
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2)
    ext = ['.JPG', '.jpg', '.png', 'PNG']
    img_paths = []
    labels = []

    for (path, dir, files) in os.walk(img_dir):
        if 'scissor' in path.split(slashs):
            for filename in files:
                if os.path.splitext(filename)[-1] in ext:
                    img_paths.append(os.path.join(path, filename))
                    labels.append(0)
        elif 'rock' in path.split(slashs):
            for filename in files:
                if os.path.splitext(filename)[-1] in ext:
                    img_paths.append(os.path.join(path, filename))
                    labels.append(1)
        elif 'paper' in path.split(slashs):
            for filename in files:
                if os.path.splitext(filename)[-1] in ext:
                    img_paths.append(os.path.join(path, filename))
                    labels.append(2)
    print("이미지 개수는 {}입니다.".format(len(img_paths)))
    if len(img_paths) == 0:
        print("이미지가 없습니다!")
        sys.exit()
    return img_paths, np.array(labels, dtype=np.int32)


def load_imgs(img_paths, norm=True):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    imgs=np.zeros(len(img_paths)*img_size[0]*img_size[1]*img_size[2],dtype=np.int32).reshape(len(img_paths),img_size[0],img_size[1],img_size[2])

    for i, img_path in enumerate(img_paths):
        img = np.array(Image.open(img_path).resize(img_size[:2], Image.ANTIALIAS), dtype=np.int32)
        if(img_size[2] == 1):
            imgs[i,:,:,:] = np.dot(img, rgb_weights).reshape(img_size[0], img_size[1], -1)
        else:
            imgs[i,:,:,:] = img
    print("data range {} ~ {}".format(np.min(imgs), np.max(imgs)))

    if norm:
        imgs = imgs/255.0
        print("-> data range {} ~ {}".format(np.min(imgs), np.max(imgs)))
    return imgs


if __name__=='__main__':
    ########################################디렉토리 트리 확인
    #Jupyter
    #local_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper"
    local_path = os.getcwd()
    slashs = '\\' if platform.system() == 'Windows' else '/'

    trees = [os.path.join(local_path, 'data')\
    , os.path.join(local_path, 'data', 'train'), os.path.join(local_path, 'data', 'test')\
    , os.path.join(local_path, 'data', 'train', 'paper'), os.path.join(local_path, 'data', 'train', 'rock'), os.path.join(local_path, 'data', 'train', 'scissor')\
    , os.path.join(local_path, 'data', 'test', 'paper'), os.path.join(local_path, 'data', 'test', 'rock'), os.path.join(local_path, 'data', 'test', 'scissor')]

    tree_result = []
    tree_result = list([ True  if  os.path.isdir(i) else False for i in trees])
    for i, _bool in enumerate(tree_result):
        if _bool == False:
            print("{}가 경로에 없습니다! 다시 만들어주세요.".format(trees[i]))
            sys.exit()

    ######################################## trainset 로드
    img_path = os.path.join(local_path, 'data', 'train')
    train_img_paths, y_train = load_path(img_path)
    x_train = load_imgs(train_img_paths, norm=True)

    ######################################## trainset 확인
    if check:
        ran_idx = random.randrange(0, len(x_train))
        if(img_size[2] == 1):
            plt.imshow(x_train[ran_idx], cmap='gray')        
        else:
            plt.imshow(x_train[ran_idx])
        print('라벨: {}'.format('가위' if y_train[ran_idx] == 0 else '바위' if y_train[ran_idx] == 1 else '보'))
        if slashs == "\\":
            plt.show()

    ######################################## testset 로드
    img_path = os.path.join(local_path, 'data', 'test')
    test_img_paths, y_test = load_path(img_path)
    x_test = load_imgs(test_img_paths, norm=True)

    ######################################## testset 확인
    if check:
        ran_idx = random.randrange(0, len(x_test))
        if(img_size[2] == 1):
            plt.imshow(x_test[ran_idx], cmap='gray')        
        else:
            plt.imshow(x_test[ran_idx])
        print('라벨: {}'.format('가위' if y_test[ran_idx] == 0 else '바위' if y_test[ran_idx] == 1 else '보'))
        if slashs == "\\":
            plt.show()

    ######################################## model 생성
    model=keras.models.Sequential()
    model.add(keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(img_size[0],img_size[1],img_size[2])))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    model.summary()
    if img_save:
        tf.keras.utils.plot_model(model, to_file=os.path.join(local_path,"imgs", "model.png"), show_shapes=True, show_layer_names=True)

    ######################################## data shuffle
    idx = np.arange(len(x_train))
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]

    ######################################## model training
    hist = model.fit(x_train, y_train, epochs=6)

    ######################################## model train result save
    if img_save:
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        loss_ax.legend(loc='upper left')

        acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
        acc_ax.set_ylabel('accuracy')
        acc_ax.legend(loc='upper right')
        plt.savefig(os.path.join(local_path,"imgs", "loss.png"))
        plt.show()
        

    ######################################## model test
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print("test_loss: {} ".format(test_loss))
    print("test_accuracy: {}".format(test_accuracy))

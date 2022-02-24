## 코드 2-1 케라스에서 MNIST 데이터셋 적재하기
# 라이브러리 불러오기 및 파일 저장
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 훈련 데이터 살펴보기
# print(train_images.shape)
# print(len(train_labels))
# print(train_labels)

# 테스트 데이터 살펴보기
# print(test_images.shape)
# print(len(test_labels))
# print(test_labels)

## 코드 2-2 신경망 구조
# 라이브러리 불러오기
from keras import models
from keras import layers

# 레이어 쌓기
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

## 코드 2-3 컴파일 단계
# 컴파일
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

## 코드 2-4 이미지 데이터 준비하기
# 훈련 이미지를 0과 1사이로 스케일 조정하기
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

# 테스트 이미지를 0과 1사이로 스케일 조정하기
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

## 레이블 준비하기
# 라이브러리 불러오기
from tensorflow.keras.utils import to_categorical

#  1차원 정수 배열을 2차원 배열로 변경하기
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

## 학습
network.fit(train_images, train_labels, epochs=5, batch_size=128)

## 결과
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
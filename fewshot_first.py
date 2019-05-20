# 0. 사용할 패키지 불러오기
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from core.base import BaseNetwork
import copy

# 랜덤시드 고정시키기
np.random.seed(3)
##################################################################################################
# 1. 데이터 생성하기

# we create two instances with the same arguments
data_gen_args = dict(rescale=1./255,
                     featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
train_input1_datagen = ImageDataGenerator(**data_gen_args)
train_input2_datagen = ImageDataGenerator(**data_gen_args)


input1_generator = train_input1_datagen.flow_from_directory(
    '../prpare_dataset/adult_sena_moma/train',
    target_size=(100, 100),
    batch_size=30,
    class_mode='categorical',
    seed=1)

input2_generator = train_input2_datagen.flow_from_directory(
    '../prpare_dataset/adult_sena_moma/train',
    target_size=(100, 100),
    batch_size=30,
    class_mode='categorical',
    seed=3)

# train_generator = zip(input1_generator, input2_generator)
#
# print(train_generator)

# train_generator = train_input1_datagen.flow_from_directory(
#         # './dataset/jjwphoto/adult/train',
#         '../prpare_dataset/adult_sena_moma/train',
#         target_size=(100, 100),
#         batch_size=30,
#         class_mode='categorical')


# def test_jjw(target):
#     print(target.batch_size)
#
#     while True:
#         batch_x, batch_y = next(target)
#         # fewshot_batch_x1 = target.flow
#         # for idx in range(1, target.batch_size):
#         #     fewshot_batch_x1.append()
#         fewshot_batch_x = [[(batch_x[0], batch_x[i])] for i in range(1, target.batch_size)]
#         fewshot_batch_y = [[int(batch_y[0][0] == batch_y[i][0]), int(batch_y[0][0] != batch_y[i][0])] for i in range(1, target.batch_size)]
#         print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         print(fewshot_batch_x)
#         print(fewshot_batch_y)
#         test = np.asarray(fewshot_batch_x)
#         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
#         print(batch_x.shape[0])
#         print('$$$$$$$$$$$$$$$$$$$$$$$$$$$')
#         print(batch_y)
#         print('###########################')
#         yield ([batch_x, batch_x], batch_y)
#         # yield (np.asarray(fewshot_batch_x), np.asarray(fewshot_batch_y))
#         # yield (batch_x, batch_y)
#
#
def test_jjw_2(target1, target2):
    print(target1.batch_size)

    while True:
        batch_x1, batch_y1 = next(target1)
        batch_x2, batch_y2 = next(target2)
        # result = copy.deepcopy(batch_y1)

        result = [[int(batch_y1[i][0] == batch_y2[i][0]), int(batch_y1[i][0] != batch_y2[i][0])] for i in
                  range(0, target1.batch_size)]
        result = np.array(result)

        yield ([batch_x1, batch_x2], result)



test_gen_args = dict(rescale=1./255)

train_test1_datagen = ImageDataGenerator(**test_gen_args)
train_test2_datagen = ImageDataGenerator(**test_gen_args)


test1_generator = train_test1_datagen.flow_from_directory(
    '../prpare_dataset/adult_sena_moma/test',
    target_size=(100, 100),
    batch_size=30,
    class_mode='categorical',
    seed=2)

test2_generator = train_test2_datagen.flow_from_directory(
    '../prpare_dataset/adult_sena_moma/test',
    target_size=(100, 100),
    batch_size=30,
    class_mode='categorical',
    seed=8)

# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(
#         # './dataset/jjwphoto/adult/test',
#         '../prpare_dataset/adult_sena_moma/test',
#         target_size=(100, 100),
#         batch_size=30,
#         class_mode='categorical')


##################################################################################################
# 2. 모델 구성하기1

# 1. RESNET
# from network.resnet import Resnet
# model_name = 'resnet.h5'
# target_network = Resnet(input_shape=(3, 100, 100), num_outputs=2, save_path=model_name)

# 2. NormalNel
# from network.normal_net import NormalNet
# model_name = 'normalnet.h5'
# target_network = NormalNet(input_shape=(100, 100, 3), num_outputs=2, save_path=model_name)

# 3. FEW SHOT
from network.fewshotnet import FewShotNet
model_name = 'fewshot.h5'
target_network = FewShotNet(input_shape=(100, 100, 3), num_outputs=2, save_path=model_name)
# target_network = FewShotNet(input_shape=(115, 115, 1), num_outputs=2, save_path=model_name)

#target_network = FewShotNet(input_shape=(3, 100, 100), num_outputs=2, save_path=model_name)


# 4. BIDIRECTIONAL LSTM
# from network.bidirectional_lstm import Bidirectional_LSTM
# model_name = 'bi_lstm.h5'
# target_network = Bidirectional_LSTM(input_shape=(3, 100, 100), num_outputs=2, save_path=model_name)



##################################################################################################
# # 3. 모델 학습시키고 저장
# target_network.train(train_data=(train_generator),
#                     # train_data=train_generator,
#                      valid_data=(test_generator),
#                      epochs=1,
#                      batch_size=1,
#                      steps_per_epoch=1,
#                      #epochs=10,
#                      #batch_size=3,
#                      #steps_per_epoch=1024,
#                      resume_training=False)


# # 3. few shot 모델 학습시키고 저장
# target_network.train(train_data=test_jjw(train_generator),
#                      valid_data=test_jjw(test_generator),
#                      epochs=1,
#                      batch_size=1,
#                      steps_per_epoch=1,
#                      #epochs=10,
#                      #batch_size=3,
#                      #steps_per_epoch=1024,
#                      resume_training=False)

# 3. few shot 모델 학습시키고 저장 input 2개
target_network.train(train_data=test_jjw_2(input1_generator, input2_generator),
                     valid_data=test_jjw_2(test1_generator, test2_generator),
                     epochs=1,
                     batch_size=1,
                     steps_per_epoch=1,
                     #epochs=10,
                     #batch_size=3,
                     #steps_per_epoch=1024,
                     resume_training=False)

target_network.predict(test1_generator, test2_generator)

# ##################################################################################################
# # 4. 모델 평가하기
# # print("-- Evaluate --")
# loaded_model = BaseNetwork.load_model('saved/models/resnet.h5')
# if loaded_model:
#     scores = BaseNetwork.evaluation(loaded_model, test_generator)
#
#
# ##################################################################################################
# # 5. 모델 사용하기
# print("-- Predict --")
# import numpy as np
# loaded_model = BaseNetwork.load_model('saved/models/resnet_34_train_sena_acc80.h5')
# if loaded_model:
#     output = BaseNetwork.predict(loaded_model, test_generator)
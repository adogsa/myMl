# 0. 사용할 패키지 불러오기
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from core.base import BaseNetwork

# 랜덤시드 고정시키기
np.random.seed(3)
##################################################################################################
# 1. 데이터 생성하기
train_datagen = ImageDataGenerator(rescale=1./255,
                                  #rotation_range=360,
                                  width_shift_range=0.01,
                                  height_shift_range=0.01,
                                  #shear_range=0.7,
                                  #zoom_range=[0.5, 2.0],
                                  horizontal_flip=True,
                                  #vertical_flip=True,
                                  #fill_mode='nearest'
                                   )

train_generator = train_datagen.flow_from_directory(
        # './dataset/jjwphoto/adult/train',
        '../prpare_dataset/adult_sena_moma/train',
        target_size=(100, 100),
        batch_size=3)

print(train_generator.class_indices)

# train_generator = train_datagen.flow_from_directory(
#         # './dataset/jjwphoto/adult/train',
#         '../prpare_dataset/adult_sena_moma/train',
#         target_size=(100, 100),
#         batch_size=30,
#         class_mode='categorical')


def test_jjw(input_gener1, input_gener2):
    print(input_gener1.batch_size)

    while True:
        batch_x, batch_y = next(input_gener1)
        batch_x_2, batch_y_2 = next(input_gener2)
        # fewshot_batch_x1 = target.flow
        # for idx in range(1, target.batch_size):
        #     fewshot_batch_x1.append()
        # fewshot_batch_x = [[(batch_x[0], batch_x[i])] for i in range(1, input_gener1.batch_size)]
        fewshot_batch_y = [[int(batch_y[i][0] == batch_y_2[i][0]), int(batch_y[i][0] != batch_y_2[i][0])] for i in range(0, input_gener1.batch_size)]
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(fewshot_batch_x)
        # print(fewshot_batch_y)
        # test = np.asarray(fewshot_batch_x)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(batch_x.shape[0])
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print(batch_y)
        print('###########################')
        yield ([batch_x, batch_x_2], fewshot_batch_y)
        # yield (np.asarray(fewshot_batch_x), np.asarray(fewshot_batch_y))
        # yield (batch_x, batch_y)


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        # './dataset/jjwphoto/adult/test',
        '../prpare_dataset/adult_sena_moma/test',
        target_size=(100, 100),
        batch_size=3)


# my_generator = train_datagen.flow_from_directory(
#         # './dataset/jjwphoto/adult/test',
#         '../prpare_dataset/adult_sena_moma/train/adult',
#         target_size=(100, 100),
#         batch_size=30,
#         class_mode=None)


##################################################################################################
# 2. 모델 구성하기1

# 1. RESNET
# from network.resnet import Resnet
# model_name = 'resnet.h5'
# target_network = Resnet(input_shape=(3, 100, 100), num_outputs=2, save_path=model_name)

# # 2. NormalNel
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
# # # # 3. 모델 학습시키고 저장
# target_network.train(train_data=train_generator,
#                     # train_data=train_generator,
#                      valid_data=test_generator,
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


# target_network.predict(test_jjw(test_generator), steps=3)

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
# loaded_model = BaseNetwork.load_model('saved/models/fewshot.h5')
# if loaded_model:
#     output = BaseNetwork.predict(loaded_model, test_generator)


# 5. 모델 사용하기
print("-- Predict --")
import numpy as np
loaded_model = target_network.load_model('saved/models/fewshot.h5')
if loaded_model:
    output = target_network.predict(test_jjw(test_generator, test_generator), steps=3)
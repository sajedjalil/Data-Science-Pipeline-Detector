vishayChaJaga = "/kaggle/input/plant-pathology-2020-fgvc7/images/"

import pandas as pd
import tensorflow as tf
import numpy as np
import cv2

kammiKaracha = tf.optimizers.Adam(learning_rate=0.01)
tf.random.set_seed(500)

yavdaVishay = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
# print(len(yavdaVishay))
from sklearn.utils import shuffle

# yavdaVishay = shuffle(yavdaVishay)
imageChaNaav = yavdaVishay['image_id']
# print(imageChaNaav.size)
yavdaLabelChaNaav = []
for i in range(imageChaNaav.__len__()):
    yavdaLabelChaNaav.append([(yavdaVishay['healthy'][i]) * 500 +
                              (yavdaVishay['multiple_diseases'][i]) * 1500 +
                              (yavdaVishay['rust'][i]) * 2000 +
                              (yavdaVishay['scab'][i]) * 2500
                              ])

pehlanchVazan = tf.Variable(tf.random.normal([3, 3, 3, 32], .00009, .00009, tf.float32, seed=1))
# dusranchVazan = tf.Variable(tf.random.normal([3, 3, 32, 64], .00009, .00009, tf.float32, seed=1))
# teesranchVazan = tf.Variable(tf.random.normal([3, 3, 64, 128], .00009, .00009, tf.float32, seed=1))
chautanchVazan = tf.Variable(tf.random.normal([3, 3, 32, 3], .00009, .00009, tf.float32, seed=1))
pachvanchVazan = tf.Variable(tf.random.normal([3, 3, 3, 3], .00009, .00009, tf.float32, seed=1))

chataanchVazan = tf.Variable(tf.random.normal([36, 1], .00009, .00009, tf.float32, seed=1))
chataanchTrshld = tf.Variable(tf.random.normal([1], .00009, .00009, tf.float32, seed=1))

# print(yavdaVishay[0:5])
# print(yavdaVishay.head())
attachaChuk = 35180072
pajeyteyChuk = 10008
print(yavdaLabelChaNaav.__len__())
while attachaChuk > pajeyteyChuk:
    for i in range((int)(yavdaLabelChaNaav.__len__() / 10)):
        if attachaChuk < pajeyteyChuk:
            break
        with tf.GradientTape() as tape:
            if attachaChuk < pajeyteyChuk:
                break
            jaraVishay = yavdaVishay[(i * 10):((i + 1) * 10)]
            # print("\n\n***********************\n---->",i,"\n",jaraVishay)
            # print(imageChaNaav[(i*10):((i+1)*10)])
            # print("\n########\n")
            # print(yavdaLabelChaNaav[(i*10):((i+1)*10)])
            # print("\n###############################################\n")

            jaraThopdi = imageChaNaav[(i * 10):((i + 1) * 10)]
            jaraNaavah = yavdaLabelChaNaav[(i * 10):((i + 1) * 10)]

            moaeeo = []
            for eachImage in jaraThopdi:
                if attachaChuk < pajeyteyChuk:
                    break
                uhotenuue = cv2.imread(vishayChaJaga + eachImage + ".jpg")
                uhotenuue = tf.cast(uhotenuue, tf.float32)
                uhotenuue = tf.reshape(uhotenuue, [1365, 2048, 3])
                # print(uhotenuue.shape)
                moaeeo.append(uhotenuue)
            moaeeo = tf.cast(moaeeo, tf.float32)
            jaraNaavah = tf.cast(jaraNaavah, tf.float32)
            # print(moaeeo.shape)

            moaeoe = tf.nn.conv2d(input=moaeeo, filters=pehlanchVazan, strides=[1, 1, 1, 1], padding='SAME')
            moaeoe = tf.nn.max_pool(input=moaeoe, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            '''
            moaeoe = tf.nn.conv2d(input=moaeoe, filters=dusranchVazan, strides=[1, 1, 1, 1], padding='SAME')
            moaeoe = tf.nn.max_pool(input=moaeoe, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            moaeoe = tf.nn.conv2d(input=moaeoe, filters=teesranchVazan, strides=[1, 1, 1, 1], padding='SAME')
            moaeoe = tf.nn.max_pool(input=moaeoe, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
            moaeoe = tf.nn.relu(moaeoe)
            '''

            moaeoe = tf.nn.conv2d(input=moaeoe, filters=chautanchVazan, strides=[1, 1, 1, 1], padding='SAME')
            moaeoe = tf.nn.max_pool(input=moaeoe, ksize=[1, 16, 16, 1], strides=[1, 16, 16, 1], padding='SAME')

            moaeoe = tf.nn.conv2d(input=moaeoe, filters=pachvanchVazan, strides=[1, 1, 1, 1], padding='SAME')
            moaeoe = tf.nn.max_pool(input=moaeoe, ksize=[1, 16, 16, 1], strides=[1, 16, 16, 1], padding='SAME')
            moaeoe = tf.nn.relu(moaeoe)

            moaeoe = tf.reshape(moaeoe, [10, 36])

            moaeoe = tf.add(tf.matmul(moaeoe, chataanchVazan), chataanchTrshld)

            loss = tf.math.reduce_mean(tf.math.square(moaeoe - jaraNaavah) + 1)
            grads = tape.gradient(loss, [pehlanchVazan, pachvanchVazan, chataanchVazan, chataanchTrshld])
            kammiKaracha.apply_gradients(zip(grads, [pehlanchVazan, pachvanchVazan, chataanchVazan, chataanchTrshld]))
            attachaChuk = loss.numpy()
            print(i, ' == ', jaraNaavah.shape, '===', moaeoe.shape, " --**-- ", attachaChuk)

bagachaYavdaVishay = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
maeoae = bagachaYavdaVishay['image_id']
tuoh = []
sntou = []
seue = []
hsteho = []

for i in range((int)(maeoae.__len__() / 10)):

    jaraThopdi = maeoae[(i * 10):((i + 1) * 10)]

    moaeeo = []
    for eachImage in jaraThopdi:
        uhotenuue = cv2.imread(vishayChaJaga + eachImage + ".jpg")
        uhotenuue = tf.cast(uhotenuue, tf.float32)
        uhotenuue = tf.reshape(uhotenuue, [1365, 2048, 3])
        # print(uhotenuue.shape)
        moaeeo.append(uhotenuue)

    moaeeo = tf.cast(moaeeo, tf.float32)

    moaeoe = tf.nn.conv2d(input=moaeeo, filters=pehlanchVazan, strides=[1, 1, 1, 1], padding='SAME')
    moaeoe = tf.nn.max_pool(input=moaeoe, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    '''
    moaeoe = tf.nn.conv2d(input=moaeoe, filters=dusranchVazan, strides=[1, 1, 1, 1], padding='SAME')
    moaeoe = tf.nn.max_pool(input=moaeoe, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    moaeoe = tf.nn.conv2d(input=moaeoe, filters=teesranchVazan, strides=[1, 1, 1, 1], padding='SAME')
    moaeoe = tf.nn.max_pool(input=moaeoe, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    moaeoe = tf.nn.relu(moaeoe)
    '''

    moaeoe = tf.nn.conv2d(input=moaeoe, filters=chautanchVazan, strides=[1, 1, 1, 1], padding='SAME')
    moaeoe = tf.nn.max_pool(input=moaeoe, ksize=[1, 16, 16, 1], strides=[1, 16, 16, 1], padding='SAME')

    moaeoe = tf.nn.conv2d(input=moaeoe, filters=pachvanchVazan, strides=[1, 1, 1, 1], padding='SAME')
    moaeoe = tf.nn.max_pool(input=moaeoe, ksize=[1, 16, 16, 1], strides=[1, 16, 16, 1], padding='SAME')
    moaeoe = tf.nn.relu(moaeoe)

    moaeoe = tf.reshape(moaeoe, [10, 36])

    moaeoe = tf.add(tf.matmul(moaeoe, chataanchVazan), chataanchTrshld)

    for j in range(10):
        if moaeoe[j] < 500:
            tuoh.append(1)
            sntou.append(0)
            seue.append(0)
            hsteho.append(0)
        elif moaeoe[j] < 1500:
            tuoh.append(0)
            sntou.append(1)
            seue.append(0)
            hsteho.append(0)
        elif moaeoe[j] < 2000:
            tuoh.append(0)
            sntou.append(0)
            seue.append(1)
            hsteho.append(0)
        elif moaeoe[j] < 2500:
            tuoh.append(0)
            sntou.append(0)
            seue.append(0)
            hsteho.append(1)

        print(i, "--", j, "---", moaeoe[j].numpy(), "-*-")

tuoh.append(0)
sntou.append(0)
seue.append(0)
hsteho.append(1)

maoeae = pd.DataFrame()

maoeae['image_id'] = maeoae
maoeae['healthy'] = tuoh
maoeae['multiple_diseases'] = sntou
maoeae['rust'] = seue
maoeae['scab'] = hsteho

maoeae.to_csv('submission.csv', index=False)

print(maoeae)

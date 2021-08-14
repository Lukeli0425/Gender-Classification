# Python人脸分类大作业 2020-8-11
# 无91 2019011559 李天骜


import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 不加这个会报错

# 1、构造数据集

# 读取下标集文件
male_file = "male_names.txt"
female_file = "female_names.txt"

male_name = []
female_name = []

with open(male_file) as male_file_object:
    for line in male_file_object:
        line = line[0:-1]  # 去除行末尾的换行符'\n'
        male_name.append(line)
np.random.shuffle(male_name)  # 随机打乱顺序

with open(female_file) as female_file_object:
    for line in female_file_object:
        line = line[0:-1]  # 去除行末尾的换行符'\n'
        female_name.append(line)
np.random.shuffle(female_name) # 随机打乱顺序
 
print(f"There are {len(male_name)} male pictures and {len(female_name)} female pictures.\n")  # 打印信息

# 将男女图片分开，分别存到两个文件夹中
train_dir = "/Users/Luke/Jupyter/Summer_Python/"  # 存放目录
for name in male_name:
    image = tf.io.read_file(f"/Users/Luke/Jupyter/Summer_Python/lfw_funneled/{name[0:-9]}/{name}")
    tf.io.write_file(f"{train_dir}male/{name}",image)
    
for name in female_name:
    image = tf.io.read_file(f"/Users/Luke/Jupyter/Summer_Python/lfw_funneled/{name[0:-9]}/{name}")
    tf.io.write_file(f"{train_dir}female/{name}",image)

# 读取图片文件，图片数据+标签的元组组成examples列表
examples = []
for name in male_name:
    image = tf.io.read_file(f"/Users/Luke/Jupyter/Summer_Python/male/{name}")
    image_tensor = tf.image.decode_jpeg(image)  # 将图片读入为tensor
    examples.append((image_tensor, 0))
male_name = []  # 清空列表
 
for name in female_name:
    image = tf.io.read_file(f"/Users/Luke/Jupyter/Summer_Python/female/{name}")
    image_tensor = tf.image.decode_jpeg(image)  # 将图片读入为tensor
    examples.append((image_tensor, 1))
female_name = []  # 清空列表

print(f"\nThere are {len(examples)} sets of data in total.\n")  # 打印信息

# 随机选择40％、10％、50％的数据，分别作为训练、验证和测试集

np.random.shuffle(examples)  # 随机打乱列表

# 取50%作为训练样例
train_data = []
train_labels = []
for (data, label) in examples[0:5000]:
    train_data.append(data)
    train_labels.append(label)
# train_labels = tf.one_hot(train_labels,depth = 2)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int32)  # 转换为张量
train_data = tf.convert_to_tensor(train_data, dtype=tf.float32)/255.0  # 转换为张量并归一化
print("\nCreate train dataset done")

# 取10%作为验证样例
valid_data = []
valid_labels = []
for (data, label) in examples[5000:6000]:
    valid_data.append(data)
    valid_labels.append(label)
# valid_labels = tf.one_hot(valid_labels,depth = 2)
valid_labels = tf.convert_to_tensor(valid_labels, dtype=tf.int32)  # 转换为张量
valid_data = tf.convert_to_tensor(valid_data, dtype=tf.float32)/255.0  # 转换为张量并归一化
print("Create valid dataset done")

# 取40%作为测试样例
test_data = []
test_labels = []
for (data, label) in examples[6000:10000]:
    test_data.append(data)
    test_labels.append(label)
# test_labels = tf.one_hot(label,depth = 2)
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int32)  # 转换为张量
test_data = tf.convert_to_tensor(test_data, dtype=tf.float32)/255.0  # 转换为张量并归一化
print("Create test dataset done\n")

examples = []  # 清空列表

# 2、构造并训练模型
# 构造CNN
print("\nConstructing CNN\n")

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(250, 250, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())  # 将三维张量展开到1维
model.add(layers.Dense(32, activation='relu'))  # 加入全连接层
model.add(layers.Dense(2))  # 输出层（0/1两种输出）

model.summary()  # 显示完整模型结构

# 编译并训练模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=10,
                    validation_data=(valid_data, valid_labels))

# 3、评估模型

# 绘制准确度曲线
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print(f"\ntest accuracy: {test_acc}\n")

# 测试集混淆矩阵
pred_test = model.predict_classes(test_data)
con_mat = confusion_matrix(test_labels, pred_test)
con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]     # 归一化
con_mat_norm = np.around(con_mat_norm, decimals=2)
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_norm, annot=True, cmap='Blues')
plt.ylim(0, 2)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title("Test data")
plt.show()

# 训练集混淆矩阵
pred_train = model.predict_classes(train_data)
con_mat = confusion_matrix(train_labels, pred_train)
con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]     # 归一化
con_mat_norm = np.around(con_mat_norm, decimals=2)
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_norm, annot=True, cmap='Blues')
plt.ylim(0, 2)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title("Train data")
plt.show()

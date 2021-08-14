# Gender-Classification
## THUEE Python Assignment 2021

### 一、设计思路
#### 1、构建数据集

根据提供的照片文件和标签文件构建数据集。

首先读入标签文件，将其中男性和女性图片的文件名分别存入male_name和female_name两个列表中，并随机打乱顺序（这里随机打乱顺序只是为了增大后面划分数据集的随机性）：
```python
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
```

考虑到提供的的照片文件都分别存放在以每个人姓名为名称的文件夹下，这在读入数据时不是很方便，因此我将所有男性和女性的图片分别转移到放在名称为male和female的文件夹内：
```python
#将男女图片分开，分别存到两个文件夹中
train_dir = "/Users/Luke/Jupyter/Summer_Python/"
for name in male_name:
    image = tf.io.read_file(f"/Users/Luke/Jupyter/Summer_Python/lfw_funneled/{name[0:-9]}/{name}")
    tf.io.write_file(f"{train_dir}male/{name}",image)
    
for name in female_name:
    image = tf.io.read_file(f"/Users/Luke/Jupyter/Summer_Python/lfw_funneled/{name[0:-9]}/{name}")
    tf.io.write_file(f"{train_dir}female/{name}",image)
```

接下来从male和female两文件夹中将jpeg文件读入为tensor，并和标签组成元组（0表示男性，1表示女性），将每组数据得到的元组存入examples列表中，并统计总共的数据组数：
```python
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
```

接下来根据examples列表中的数据（共13234组），随机选取5000组作为训练数据集，1000组作为验证数据集，4000组作为测试数据集：
```python
np.random.shuffle(examples)  # 随机打乱列表

# 取50%作为训练样例
train_data = []
train_labels = []
for (data, label) in examples[0:5000]:
    train_data.append(data)
    train_labels.append(label)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int32)  # 转换为张量
train_data = tf.convert_to_tensor(train_data, dtype=tf.float32)/255.0  # 转换为张量并归一化
print("\nCreate train dataset done")

# 取10%作为验证样例
valid_data = []
valid_labels = []
for (data, label) in examples[5000:6000]:
    valid_data.append(data)
    valid_labels.append(label)
valid_labels = tf.convert_to_tensor(valid_labels, dtype=tf.int32)  # 转换为张量
valid_data = tf.convert_to_tensor(valid_data, dtype=tf.float32)/255.0  # 转换为张量并归一化
print("Create valid dataset done")

# 取40%作为测试样例
test_data = []
test_labels = []
for (data, label) in examples[6000:10000]:
    test_data.append(data)
    test_labels.append(label)
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int32)  # 转换为张量
test_data = tf.convert_to_tensor(test_data, dtype=tf.float32)/255.0  # 转换为张量并归一化
print("Create test dataset done\n")

examples = []  # 清空列表
```

#### 2、构造与训练模型

考虑到本次的任务是分类人脸，而对于人脸的操作主要涉及到对图片某一区域特征的提取，因此我选择使用卷积神经网络作为模型。为了达到一定的准确率，模型的网络层数不宜过少，但考虑到训练效率也不宜过大。在最后需要采用onehot编码的方式输出判断结果，因此在最后也需要加入一个或多个全连接层，其中最后一层就是输出层（我们的问题中只有男/女（0/1）两类，因此输出层只需要两个输出）。最终我的模型的具体构造方案如下：加入4个Conv2D层和3个MaxPooling2D层，并在最后加入两个全连接层（包括输出层）：
```python
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
```

使用complie函数和fit函数编译并训练模型（选择epoch为重复10次）：
```python
# 编译并训练模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=10,
                    validation_data=(valid_data, valid_labels))
```

#### 3、评估模型

首先利用matplotlib画出模型判断准确率（训练集上的准确率和验证集上的准确率）随epoch变化的曲线：
```python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
```

接下来将测试数据集输入训练好的模型，显示性别判断准确率：
```python
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print(f"\ntest accuracy: {test_acc}\n")
```

利用sklearn画出模型在训练集和测试集上的混淆矩阵，将模型性能可视化：
```python
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
```

### 二、运行结果
#### 1、第一次训练

从13234组数据中随机选取10000组（5000组用于训练，1000组用于验证，4000组用于测试）进行训练，得到的控制台输出：
```
There are 10268 male pictures and 2966 female pictures.

There are 13234 sets of data in total.

Create train dataset done
Create valid dataset done
Create test dataset done

Constructing CNN
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 248, 248, 32)      896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 124, 124, 32)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 122, 122, 64)      18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 61, 61, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 59, 59, 64)        36928     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 29, 29, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 27, 27, 64)        36928     
_________________________________________________________________
flatten (Flatten)            (None, 46656)             0         
_________________________________________________________________
dense (Dense)                (None, 32)                1493024   
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 66        
=================================================================
Total params: 1,586,338
Trainable params: 1,586,338
Non-trainable params: 0
_________________________________________________________________
Train on 5000 samples, validate on 1000 samples
Epoch 1/10
5000/5000 [==============================] - 631s 126ms/sample - loss: 0.4187 - accuracy: 0.8100 - val_loss: 0.3738 - val_accuracy: 0.8490
Epoch 2/10
5000/5000 [==============================] - 375s 75ms/sample - loss: 0.3239 - accuracy: 0.8588 - val_loss: 0.2722 - val_accuracy: 0.8770
Epoch 3/10
5000/5000 [==============================] - 357s 71ms/sample - loss: 0.2536 - accuracy: 0.8926 - val_loss: 0.3295 - val_accuracy: 0.8650
Epoch 4/10
5000/5000 [==============================] - 349s 70ms/sample - loss: 0.1939 - accuracy: 0.9212 - val_loss: 0.2705 - val_accuracy: 0.9000
Epoch 5/10
5000/5000 [==============================] - 338s 68ms/sample - loss: 0.1594 - accuracy: 0.9378 - val_loss: 0.2658 - val_accuracy: 0.9110
Epoch 6/10
5000/5000 [==============================] - 345s 69ms/sample - loss: 0.1082 - accuracy: 0.9570 - val_loss: 0.2897 - val_accuracy: 0.9010
Epoch 7/10
5000/5000 [==============================] - 338s 68ms/sample - loss: 0.0731 - accuracy: 0.9712 - val_loss: 0.2721 - val_accuracy: 0.9130
Epoch 8/10
5000/5000 [==============================] - 334s 67ms/sample - loss: 0.0460 - accuracy: 0.9816 - val_loss: 0.2922 - val_accuracy: 0.9150
Epoch 9/10
5000/5000 [==============================] - 338s 68ms/sample - loss: 0.0359 - accuracy: 0.9880 - val_loss: 0.5020 - val_accuracy: 0.8760
Epoch 10/10
5000/5000 [==============================] - 345s 69ms/sample - loss: 0.0513 - accuracy: 0.9816 - val_loss: 0.3779 - val_accuracy: 0.9140

4000/1 - 150s - loss: 0.2890 - accuracy: 0.9115

test accuracy: 0.9114999771118164
```

模型准确率随epoch变化曲线：
![](https://github.com/Lukeli0425/Gender-Classification/raw/main/test1_curve.jpg)

训练数据集的混淆矩阵：
![](https://github.com/Lukeli0425/Gender-Classification/raw/main/test1_train_mat.jpg)

测试数据集的混淆矩阵：
![](https://github.com/Lukeli0425/Gender-Classification/raw/main/test1_test_mat.jpg)

从结果中我发现，虽然模型在训练集上的准确率始终在增大，并在最后逼近100%，但是验证集上的准确率并没有相似的变化趋势，而是时增时减，总体在相对缓慢地增大。同时，即使训练集准确率很高，测试集上的准确率也只有90%左右，这说明得到的模型是更“适用于”训练集的模型。画出混淆矩阵后，我发现虽然训练集对于男女样例的准确率都很高，但对于测试集的男性准确率（94%）远高于女性（80%），我猜想这是由于数据集中男性样本远多于女性所致，即对于男性照片的判断已经被训练得很完善，但是对于女性照片的判读由于训练样本较少而不太完善。为了解决这个问题，我又进行了一次尝试。

#### 2、第二次训练

为了使得男女两种标签的训练样例数量大致相等，我从男性照片中随机选择3034张，并选取全部2966张女性照片，组成由6000张照片组成的数据集（其中3600组用于训练，600组用于验证，2400组用于测试），这样男女样本数量相差无几。重新对模型进行训练，得到的控制台输出：
```
There are 3034 male pictures and 2966 female pictures.

There are 6000 sets of data in total.

train data done
valid data done
test data done

Constructing CNN
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 248, 248, 32)      896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 124, 124, 32)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 122, 122, 64)      18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 61, 61, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 59, 59, 64)        36928     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 29, 29, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 27, 27, 64)        36928     
_________________________________________________________________
flatten (Flatten)            (None, 46656)             0         
_________________________________________________________________
dense (Dense)                (None, 32)                1493024   
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 66        
=================================================================
Total params: 1,586,338
Trainable params: 1,586,338
Non-trainable params: 0
_________________________________________________________________
Train on 3000 samples, validate on 600 samples
Epoch 1/10
3000/3000 [==============================] - 230s 77ms/sample - loss: 0.5231 - accuracy: 0.7230 - val_loss: 0.4297 - val_accuracy: 0.8067
Epoch 2/10
3000/3000 [==============================] - 214s 71ms/sample - loss: 0.3780 - accuracy: 0.8323 - val_loss: 0.3788 - val_accuracy: 0.8150
Epoch 3/10
3000/3000 [==============================] - 207s 69ms/sample - loss: 0.2873 - accuracy: 0.8867 - val_loss: 0.4650 - val_accuracy: 0.7833
Epoch 4/10
3000/3000 [==============================] - 206s 69ms/sample - loss: 0.2295 - accuracy: 0.9113 - val_loss: 0.3238 - val_accuracy: 0.8700
Epoch 5/10
3000/3000 [==============================] - 209s 70ms/sample - loss: 0.1599 - accuracy: 0.9387 - val_loss: 0.3588 - val_accuracy: 0.8767
Epoch 6/10
3000/3000 [==============================] - 207s 69ms/sample - loss: 0.1133 - accuracy: 0.9533 - val_loss: 0.4284 - val_accuracy: 0.8750
Epoch 7/10
3000/3000 [==============================] - 207s 69ms/sample - loss: 0.0865 - accuracy: 0.9663 - val_loss: 0.4765 - val_accuracy: 0.8433
Epoch 8/10
3000/3000 [==============================] - 205s 68ms/sample - loss: 0.0676 - accuracy: 0.9773 - val_loss: 0.6500 - val_accuracy: 0.8433
Epoch 9/10
3000/3000 [==============================] - 204s 68ms/sample - loss: 0.0287 - accuracy: 0.9883 - val_loss: 0.5796 - val_accuracy: 0.8683
Epoch 10/10
3000/3000 [==============================] - 204s 68ms/sample - loss: 0.0336 - accuracy: 0.9877 - val_loss: 0.6176 - val_accuracy: 0.8517

2400/1 - 80s - loss: 0.5415 - accuracy: 0.8683

test accuracy: 0.8683333396911621
```

模型准确率随epoch变化曲线：
![](https://github.com/Lukeli0425/Gender-Classification/raw/main/test2_curve.jpg)

训练数据集的混淆矩阵：
![](https://github.com/Lukeli0425/Gender-Classification/raw/main/test2_train_mat.jpg)

测试数据集的混淆矩阵：
![](https://github.com/Lukeli0425/Gender-Classification/raw/main/test2_test_mat.jpg)

这次得到的模型在训练集上的准确率仍旧达到了99%以上，在测试集上对于男性和女性两种照片的判断正确率相差无几。虽然男性正确率相比第一次模型的有所降低，但是女性正确率显著提高，整体准确率依旧达到了87%。我们得到了一个对于不同标签的样例都有较好准确率的模型。

### 三、小结

本次作业可谓是我的深度学习初体验，我利用卷积神经网络完成了一个简单的人脸性别分类问题。这次作业给我提供了一个了解深度学习基本原理并简单实践的机会，同时加强了我对python的掌握。我在业中用到了Tensorflow，Sklearn等深度学习框架，亲身体会到了运用官方文档中的信息完成自己的任务的过程，熟悉了这些深度学习框架的使用，并且也练习了python的文件操作，强化了我对python这门语言的掌握，也提升了我自己寻找信息解决问题的能力。

完成本次作业的过程并非一帆风顺。首先，我在网上寻找了大量相关资料并尝试将一些方法用到我的代码中来，其中遇到了很多问题，有些是对框架中函数使用方法的不熟悉，也有对深度学习理论的不了解。在攻克这些代码实现的问题之后，我在训练模型的过程中也发现了一些问题。比如说最初的模型由于层数太少，模型性能很不好，于是我增大网络层数；之后又发现提供的都照片中男性多于女性，导致训练得到的模型对于男性的判断正确率很高，但是女性判断准确率较低，于是我又对数据集进行调整，使得男女样本数量差不多，这才得到了一个对于标签更“平衡”的模型。后来我有发现了将模型性能利用混淆矩阵可视化的方法，就进一步改善我的模型评估方法。这些不断发现问题并解决问题的过程非常宝贵，在逐一解决这些问题之后，我的信心得到了很大的增强，也不再怵遇到一些奇奇怪怪的问题了。

虽然本次作业比较难，但受益颇多。

### 四、文件清单

    README.md                                       说明文档
    male_names.txt                                  男性照片名称文件
    female_names.txt                                女性照片名称文件
    lfw_funneled.tar                                全部照片（后附下载链接）
    gender-classification.py                        完整代码
    test1_curve.jpg                                 第一次训练模型的准确率变化曲线
    test1_test_mat.jpg                              第一次训练模型的测试集混淆矩阵
    test1_train_mat.jpg                             第一次训练模型的训练集混淆矩阵
    test2_curve.jpg                                 第二次训练模型的准确率变化曲线
    test2_test_mat.jpg                              第二次训练模型的测试集混淆矩阵
    test2_train_mat.jpg                             第二次训练模型的训练集混淆矩阵
    2021年夏季学期Python程序设计大作业指引0721.pdf       指导文档

[照片下载链接](https://cloud.tsinghua.edu.cn/f/71a2146a09f24b9984a5/)

### 五、参考资料

[Python编程：从入门到实践](https://www.ituring.com.cn/book/1861)

[TensorFlow官方文档](https://tensorflow.google.cn/tutorials?hl=zh_cn)

[Scikit-Learn官方文档](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)


---

#### Contact me: lukeli@sina.cn  lta19@mails.tsinghua.edu.cn
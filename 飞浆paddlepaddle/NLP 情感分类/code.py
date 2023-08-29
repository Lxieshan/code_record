# 导入必要的包
import paddle

import paddle.dataset.imdb as imdb

import paddle.fluid as fluid

import numpy as np

import os

!mkdir -p /home/aistudio/.cache/paddle/dataset/imdb/

!cp /home/aistudio/data/data7990/aclImdb_v1.tar.gz /home/aistudio/.cache/paddle/dataset/imdb/

# 获取数据字典

print("加载数据字典中...")

word_dict = imdb.word_dict()

# 获取数据字典长度

dict_dim = len(word_dict)

print('完成')

数据是以数据标签的方式表示一个句子。

所以每个句子都是以一串整数来表示的，每个数字都是对应一个单词。

数据集就会有一个数据集字典，这个字典是训练数据中出现单词对应的数字标签。


# 获取训练和预测数据

print("加载训练数据中...")

train_reader = paddle.batch(paddle.reader.shuffle(imdb.train(word_dict),

                                                  512),

                            batch_size=128)

print("加载测试数据中...")

test_reader = paddle.batch(imdb.test(word_dict), 

                           batch_size=128)

print('完成')


# 定义长短期记忆网络

def lstm_net(ipt, input_dim):

    # 以数据的IDs作为输入

    emb = fluid.layers.embedding(input=ipt, size=[input_dim, 128], is_sparse=True)

    # 第一个全连接层

    fc1 = fluid.layers.fc(input=emb, size=128)

    # 进行一个长短期记忆操作

    lstm1, _ = fluid.layers.dynamic_lstm(input=fc1, #返回：隐藏状态（hidden state），LSTM的神经元状态

                                         size=128) #size=4*hidden_size

    # 第一个最大序列池操作

    fc2 = fluid.layers.sequence_pool(input=fc1, pool_type='max')

    # 第二个最大序列池操作

    lstm2 = fluid.layers.sequence_pool(input=lstm1, pool_type='max')

    # 以softmax作为全连接的输出层，大小为2,也就是正负面

    out = fluid.layers.fc(input=[fc2, lstm2], size=2, act='softmax')

    return out

# 这里可以先定义一个输入层，这样要注意的是我们使用的数据属于序列数据，所以我们可以设置lod_level为1，当该参数不为0时，表示输入的数据为序列数据，默认lod_level的值是0.

# 定义输入数据， lod_level不为0指定输入数据为序列数据

words = fluid.layers.data(name='words', shape=[1], dtype='int64', lod_level=1)

label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取长短期记忆网络

model = lstm_net(words, dict_dim)

# 接着定义损失函数，这里同样是一个分类任务，所以使用的损失函数也是交叉熵损失函数。这里也可以使用fluid.layers.accuracy()接口定义一个输出分类准确率的函数，可以方便在训练的时候，输出测试时的分类准确率，观察模型收敛的情况。

# 获取损失函数和准确率

cost = fluid.layers.cross_entropy(input=model, label=label)

avg_cost = fluid.layers.mean(cost)

acc = fluid.layers.accuracy(input=model, label=label)

# 获取预测程序

test_program = fluid.default_main_program().clone(for_test=True)

# 然后是定义优化方法，这里使用的时Adagrad优化方法，Adagrad优化方法多用于处理稀疏数据，设置学习率为0.002。

# 定义优化方法

optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.002)

opt = optimizer.minimize(avg_cost)

# 如果读取有GPU环境，可以尝试使用GPU来训练，使用方式是使用fluid.CUDAPlace(0)来创建。

# 定义使用CPU还是GPU，使用CPU时use_cuda = False,使用GPU时use_cuda = True
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义数据数据的维度，数据的顺序是一条句子数据对应一个标签。
# 定义输入数据的维度

# 定义数据数据的维度，数据的顺序是一条句子数据对应一个标签

feeder = fluid.DataFeeder(place=place, feed_list=[words, label])

# 现在就可以开始训练了，这里设置训练的循环是2次，大家可以根据情况设置更多的训练轮数。我们在训练中，每40个Batch打印一层训练信息和进行一次测试，测试是使用测试集进行预测并输出损失值和准确率，测试完成之后，对之前预测的结果进行求平均值。
# 开始训练

for pass_id in range(1):

    # 进行训练

    train_cost = 0

    for batch_id, data in enumerate(train_reader()):              #遍历train_reader迭代器

        train_cost = exe.run(program=fluid.default_main_program(),#运行主程序

                             feed=feeder.feed(data),              #喂入一个batch的数据

                             fetch_list=[avg_cost])               #fetch均方误差


        if batch_id % 40 == 0:                 #每40次batch打印一次训练、进行一次测试

            print('Pass:%d, Batch:%d, Cost:%0.5f' % (pass_id, batch_id, train_cost[0]))

    # 进行测试

    test_costs = []   #测试的损失值

    test_accs = []    #测试的准确率

    for batch_id, data in enumerate(test_reader()):

        test_cost, test_acc = exe.run(program=test_program,

                                            feed=feeder.feed(data),

                                             fetch_list=[avg_cost, acc])

        test_costs.append(test_cost[0])

        test_accs.append(test_acc[0])

    # 计算平均预测损失在和准确率

    test_cost = (sum(test_costs) / len(test_costs))

    test_acc = (sum(test_accs) / len(test_accs))

    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))

#保存模型

model_save_dir = "/home/aistudio/work/emotionclassify.inference.model"

# 如果保存路径不存在就创建

if not os.path.exists(model_save_dir):

    os.makedirs(model_save_dir)

print ('save models to %s' % (model_save_dir))

fluid.io.save_inference_model(model_save_dir, #保存推理model的路径

                                  ['words'],      #推理（inference）需要 feed 的数据

                                  [model],         #保存推理（inference）结果的 Variables

                                  exe)            #exe 保存 inference model

# 我们先定义三个句子，第一句是中性的，第二句偏向正面，第三句偏向负面。然后把这些句子读取到一个列表中。

# 定义预测数据

reviews_str = ['read the book forget the movie', 'this is a great movie', 'this is very bad']

# 把每个句子拆成一个个单词

reviews = [c.split() for c in reviews_str]


# 然后把句子转换成编码，根据数据集的字典，把句子中的单词转换成对应标签。

# 获取结束符号的标签

UNK = word_dict['<unk>']

# 获取每句话对应的标签

lod = []

for c in reviews:

    # 需要把单词进行字符串编码转换

    lod.append([word_dict.get(words.encode('utf-8'), UNK) for words in c])

# 获取输入数据的维度和大小。
# 获取每句话的单词数量

base_shape = [[len(c) for c in lod]]

# 将要预测的数据转换成张量，准备开始预测。

# 生成预测数据

tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
infer_exe = fluid.Executor(place)    #创建推测用的executor

inference_scope = fluid.core.Scope() #Scope指定作用域

with fluid.scope_guard(inference_scope):#修改全局/默认作用域（scope）, 运行时中的所有变量都将分配给新的scope。

    #从指定目录中加载 推理model(inference model)

    [inference_program,                                            #推理的program

     feed_target_names,                                            #str列表，包含需要在推理program中提供数据的变量名称

     fetch_targets] = fluid.io.load_inference_model(model_save_dir,#fetch_targets: 推断结果，model_save_dir:模型训练路径 

                                                        infer_exe) #infer_exe: 运行 inference model的 executor

    results = infer_exe.run(inference_program,                                 #运行预测程序

                            feed={feed_target_names[0]: tensor_words},#喂入要预测的x值

                            fetch_list=fetch_targets)                           #得到推测结果 

    # 打印每句话的正负面概率

    for i, r in enumerate(results[0]):

        print("\'%s\'的预测结果为：正面概率为：%0.5f，负面概率为：%0.5f" % (reviews_str[i], r[0], r[1]))



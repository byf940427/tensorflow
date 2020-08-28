###莫烦Python代码讲解###

'''lesson 1
代码目的： 学习并预测y = 0.1x+0.3 中的斜率与偏离值（0.1，0.3）
'''

### 代码收获：   代码思路--建立基础变量，建立训练变量，初始化数据，创建session
### 需要创建weights，biases和方程y用于学习
### 训练过程中需要变量：损失loss，优化器optimizer，和训练基准train，train = optimizer.minimize(loss)意味着使用优化器使loss值最小，得到最优解
### 一定要使用init = tf.initialize_all_variable()初始化，在此之前虽然定义了变量但不初始化无法使用
### 具体的过程需要在tf.Session中运行

import tensorflow as tf 
import numpy as np 

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0)) # tf.Variable 参数变量, 一维从-1到1的变量
bias = tf.Variable(tf.zeros([1])) # 一维的0

y = Weights * x_data + bias

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5) # 学习效率0.5
train = optimizer.minimize(loss)

init = tf.initialize_all_variables() # 初始化所有变量，虽然前面设定变量了，但没initialize就无法使用

sess = tf.Session()
sess.run(init) # 开始运行

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights),sess.run(bias)) # 每20次输出训练得到的Weights和bias的值

### create tensorflow structure end ###




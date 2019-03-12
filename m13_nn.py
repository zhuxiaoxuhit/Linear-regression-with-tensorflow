import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
def add_layer(inputs,in_size,out_size,activation_function = None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]))
            tf.summary.histogram('/weights',Weights)#生成weight的histogram
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]))
            tf.summary.histogram('/biases',biases)
        with tf.name_scope('Wx_plus_b'):     
            Wx_plus_b = tf.matmul(inputs,Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram('/outputs',outputs)
        return outputs



#数据集  x在-1到1之间取300个点，y = x^2 + 0.5 + noise 
x_data = np.linspace(-1,1,300)[:,np.newaxis]  #每次训练输入的元素,生成[[],[],[],....[]]这样的tensor,里面300个元素
noise = np.random.normal(0,0.05,x_data.shape) #生成正太的随机数tensor,均值0,标准差0.05
y_data = np.square(x_data) - 0.5 + noise      #训练集给出的标准答案,用于修正Weights 和 biases
#用来接收x_data和y_data 
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name = 'x_input')
    ys = tf.placeholder(tf.float32,[None,1],name = 'y_input')


#第一层神经网络,输入是x_data,0个输入神经元,输出10个神经元,relu的激励函数
l1 = add_layer(xs,1,10,activation_function= tf.nn.relu)
#输出层神经网络 ,输入是上一层的输出,10个神经元,输出是预测数值
prediction = add_layer(l1,10,1,activation_function = None)


#loss 是300个数据集输入组成的tensor经过计算得到的300个prediction和y_data之间的误差的和的平均值,结果是一个数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices = [1]))
    #loss是纯量(scalar),在tensorboard的events里面显示
    tf.summary.scalar('loss',loss)
#每次训练用梯度下降的方法优化loss(最小化) 
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


sess = tf.Session()
#合并所有的summary
merged = tf.summary.merge_all()
#注意这里版本更改了,不是之前的tf.train.SummaryWriter()了
writer = tf.summary.FileWriter('/home/zhuxiaoxu/Desktop/logs',sess.graph)#把整个框架加载到指定的文件里,然后在终端通过tensorboard --logdir /home/zhuxiaoxu/Desktop/logs得到网址
#初始化所有变量
init = tf.global_variables_initializer()
###记得initial所有变量
sess.run(init)
for i in range(1000):
    sess.run(train_step,feed_dict = {xs:x_data,ys:y_data})
    if i%50 == 0:
        result = sess.run(merged,feed_dict = {xs:x_data,ys:y_data})#tensorboard上的Summary也要run一下 
        writer.add_summary(result,i)


'''
未成功!
fig = plt.figure()#生成图片框
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data) #以点的形式画
plot.show()
'''



'''
with tf.Session() as sess:
    sess.run(init)
    #训练1000次,共1000个tensor,1000*300个输入数据
    #为什么要用placeholder()的方式,因为我们的数据集是多次输入,也就是每训练一次,就要生成300个随机输入值
    #也就是训练过程中通过placeholder()多次得到随机的数据集
    for i in range(1000):
        sess.run(step_train,feed_dict = {xs:x_data,ys:y_data})
        if i %50 == 0:
            print(sess.run(loss,feed_dict = {xs:x_data, ys:y_data}))


'''


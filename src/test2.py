import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.preprocessing import MinMaxScaler

tf.disable_v2_behavior()
def linear_regression2(dt,ds):
    
    #정규화를 위해 numpy배열로 변경하여 형태를 변경해줌
    x_train = np.array(dt)
    y_train = np.array(ds)
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    
    #기울기 표현을 위한 차트 그리는 부분
    #plt.plot(x_train, y_train)
    #plt.show()
    
    #실제 정규화부분(정규화를 위해 MinMaxScaler사용)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_train = x_scaler.fit_transform(x_train)
    
    #한쪽의 차원에 맞게 정규화를 진행
    y_train = x_scaler.transform(y_train)
    
    W=tf.Variable(tf.random_normal([1]), name="weight")
    b=tf.Variable(tf.random_normal([1]), name="bias")

    hypothesis = x_train*W+b
    cost = tf.reduce_mean(tf.square(hypothesis-y_train))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001,momentum=0.9)
    train = optimizer.minimize(cost)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(501):
        sess.run(train)
        if step%500 == 0:
            print(step,sess.run(cost), sess.run(W), x_scaler.inverse_transform(sess.run(b).reshape(-1,1)))
    #print(sess.run(W))
    #print()
    return sess.run(W), np.mean(sess.run(W))


def linear_regression(dt, ds):
    tf.disable_v2_behavior()

    W = tf.Variable(tf.random_normal([1]))
    b = tf.Variable(tf.random_normal([1]))

    X = tf.placeholder(tf.float32, shape=[None])
    Y = tf.placeholder(tf.float32, shape=[None])

    hypothesis = X * W + b

    cost = tf.reduce_mean(tf.square(hypothesis-Y))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.000000005)
    train = optimizer.minimize(cost)
    pattern = []
    pred = []
    costt = []
    sess = tf.Session()

    for i in range(len(dt)):
        sess.run(tf.global_variables_initializer())
        tempcost = []
        
        for step in range(501):
            _, cost_val, W_val, b_val = sess.run([train, cost, W, b],feed_dict={X:dt[i], Y:ds[i]})
            tempcost.append(W_val)
        
        print("//")
        if math.isnan(W_val[0]):
            print("raise nan")
            continue
        print(step, W_val, cost_val)
        pattern.append(W_val[0])
        pred.append(W_val*ds[i] + b_val)
        costt.append(tempcost)

    #delta seq no 평균을 구한다.
    print("Delta Seq No : {}".format(np.mean(pattern)))
    
    return pattern, np.mean(pattern)


a = str("0x00000001")
print(int(a,16))
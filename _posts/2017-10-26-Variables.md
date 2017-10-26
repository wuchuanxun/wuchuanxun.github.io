---
layout: post
title: Variables
categories: [Tensorflow]
description:  The difference between scope, variable
keywords: scope, variable
---

你是否也对于TensorFlow的变量管理感到疑惑？当我们搭建一个很大的网络的时候，我们有太多的变量，给每一个变量设置一个不同的名字是一个很辛苦的活，而且有些如果为了避免变量名称冲突而选择一个毫无意义的名称会让后面回顾的时候很麻烦。所以TensorFlow为我们设置了一套变量的机制，我们可以灵活的使用name_space 和 variable_scope，他们有什么不一样呢？

## name_scope的使用

我们先来看个例子：

```pyhton
import tensorflow as tf

with tf.name_scope("a_name_scope"):
    initializer = tf.constant_initializer(value=1)
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, 		initializer=initializer)
    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
    var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)
    varsum = var21+var22

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(var1.name)        # var1:0
    print(sess.run(var1))   # [ 1.]
    print(var2.name)        # a_name_scope/var2:0
    print(sess.run(var2))   # [ 2.]
    print(var21.name)       # a_name_scope/var2_1:0
    print(sess.run(var21))  # [ 2.0999999]
    print(var22.name)       # a_name_scope/var2_2:0
    print(sess.run(var22))  # [ 2.20000005]
    print(varsum.op.name)   #a_name_scope/add
    print(sess.run(varsum)) # [ 4.30000019]
```

我们可以看到name_scope可以建立一个名称管理空间，可以规范名称空间内部的variable和tensor，如果利用tf.Variable()建立两个名称一样的变量，TensorFlow会自动加入不同的子下标以示区分。而如果我们采用tf.get_variable()来获取变量，则变量的名称不受name_scope的限制。



## variable_scope的使用

同样我们以例子来看：

```python
with tf.name_scope('second'):
    with tf.variable_scope("a_variable_scope") as scope:
        initializer = tf.constant_initializer(value=3)
        var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer)
        scope.reuse_variables()
        var3_reuse = tf.get_variable(name='var3',)
        var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
        var4_reuse = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
        varsum = var4+var4_reuse
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var3.name)            # a_variable_scope/var3:0
    print(sess.run(var3))       # [ 3.]
    print(var3_reuse.name)      # a_variable_scope/var3:0
    print(sess.run(var3_reuse)) # [ 3.]
    print(var4.name)            # a_variable_scope/var4:0
    print(sess.run(var4))       # [ 4.]
    print(var4_reuse.name)      # a_variable_scope/var4_1:0
    print(sess.run(var4_reuse)) # [ 4.]
    print(varsum.op.name)       # second/a_variable_scope/add
    print(sess.run(varsum))     # [ 8.]
```

可以看出variable_scope同样限制了所有的variable和tensor，不仅如此，使用tf.get_variable()也要受到variable_scope的约束。同时我们还可以注意到，如果要拥有C++语言一样的引用(或者说全局变量一样的变量重用机制)，get_variable()实现了这个功能。



## get_variable实例

这个实例综合了scope的分别和变量重用：

```python
class TestClass(object):
    def __init__(self):
   		self.a=tf.get_variable('a_value',shape=[3],initializer=tf.constant_initializer(1)) #没用scope
        with tf.name_scope('env1'):
            self.b=tf.get_variable('b_value',shape=[3],initializer=tf.constant_initializer(0)) #使用name_scope
            self.aplusb=self.a+self.b		#operation,get tensor
        with tf.variable_scope('env2'):
            self.c=tf.get_variable('c_value',shape=[3],initializer=tf.constant_initializer(-1))#使用variable_scope

disp=lambda x:print(x)
with tf.Session() as sess:
    with tf.name_scope('Test4'):
        T4=TestClass()
    with tf.variable_scope('Test0',reuse=False):
        T1=TestClass()
    with tf.variable_scope('Test1'):
        T2=TestClass()
    with tf.variable_scope('Test1',reuse=True):
        T3=TestClass()
    sess.run(tf.global_variables_initializer())
    disp(sess.run(T3.a))
    sess.run(T2.a.assign(T1.a+3))	#[ 1.  1.  1.]
    disp(sess.run(T3.a))			#[ 4.  4.  4.]
    disp(sess.run(T2.a))			#[ 4.  4.  4.]
    disp(T1.a.name)					#Test0/a_value:0
    disp(T1.b.name)					#Test0/b_value:0
    disp(T1.c.name)					#Test0/env2/c_value:0
    disp(T4.a.name)					#a_value:0
    disp(T4.b.name)					#b_value:0
    disp(T4.c.name)					#env2/c_value:0
    disp(T1.aplusb.op.name)			#Test0/env1/add
    disp(T2.aplusb.op.name)			#Test1/env1/add
    disp(T3.aplusb.op.name)			#Test1_1/env1/add
    disp(sess.run(T3.aplusb)) 		#[ 4.  4.  4.]
    sess.run(T2.a.assign(T1.a+30))
    disp(sess.run(T3.aplusb))		#[ 31.  31.  31.]
    disp(sess.run(T2.aplusb))		#[ 31.  31.  31.]
    tf.summary.FileWriter('./log',sess.graph)	#观察tensor graph
```

可以看出get_variable可以共用变量，不过在引用相同名称变量的时候，需要在variable_scope下设置reuse=True，如果是创建一个全新的变量，reuse需要设置为False或者None（默认）。在给变量赋值的时候需要使用assign操作，否则变量就不是原来的变量，相当于新创建一个变量。我们可以看到name_scope是不受variable reuse 限制的（这是一个命名空间，不是变量），所以出现同名的name_scope的时候也会自动加以区分。

为了更好看到类成员之间的关系，可以详细观看graph：

<p align="center">
<img src="/images/Tensorflow/TensorBoard/variable_diff1.PNG"/>
</p>

可以看到a,b不属于env1这个name_scope(因为用get_variable得到变量)，但是c同样也是get_variable得到，而包含于variable_scope(env2)，我们着重看一下Test1里面的变量共用机制：

<p align="center">
<img src="/images/Tensorflow/TensorBoard/variable_diff2.PNG" style="zoom:70%"/>
</p>

可以看到变量的确是共用的，只有这个add操作是不一样的（不同的name_scope)。
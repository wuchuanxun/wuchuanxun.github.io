---
layout: post
title: About Tensor shape
categories: [Tensorflow]
description: The input of some function
keywords: Tensor Shape
---

Here is a function commonly used:	
```python
tf.nn.sparse_softmax_cross_entropy_with_logits(labels,logits)
```
The labels is with type tf.int32 or tf.int64 and it's rank is one less than logits. e.g: labels is in the shape of [d1,d2,...,dr], then the shape of logits should be [d1,d2,...,dr,n_class]. And the function return the tensor with the same shape of labels

```python
tf.placeholder(shape=None)
```

The scalar has shape () , even you create a array by np.array(scalar), it still with the shape (), if you want to create a shape [None], you should use np.array(scalar).reshape([-1]), which will return a array with shape [1,] (same as [1]). 

In placeholder, if use None(default) as shape, you can feed any shape to the placeholder, if you want specify some dimensions, you can use shape as follows: [None,3],[None,None,4]

About tf.add and tf.multipy, this is element-wise operation.  Note the two tensor in operation is a and b, assume ran(a)>=rank(b).  suppose a with the shape [s1,s2,...,sn], then the b can with the shape [1],[1 or s1,1 or s2,...,1 or sn],[sk,...,sn]


---
layout: post
title: Multiprocessing
categories: [python]
description: 多进程，共享变量
keywords: Multiprocessing,python
---
之前提到python在处理复杂任务的时候，为了提高效率可以考虑多线程。但是由于多线程的诸多限制，很多情况下CPU密集型的任务使用多线程反而会降低效率，所以解决方案是使用多进程。进程是操作系统中最基本的概念，它是一个实体。每一个进程都有自己的地址空间(文本区，数据区，堆栈)，是一个执行实例。下面我们就来了解python的多进程实现和变量共享的实现。

## multiprocessing 模块

multiprocessing 模块为在**子进程**中运行任务、通信、共享数据，以及执行各种形式的同步提供支持。该模块更适合在UNIX下使用。与线程不同的是，**进程没有共享状态，所以数据修改是相互独立的，不影响其它的进程**，有时候这是个优点，但是很多时候也会带来不便。

### Process

先来看一个函数：`P=Process(group=None, target=None, name=None, args=(), kwargs={})` 

参数以及属性如下：

```
group: 预留参数	target：进程的可调用对象	name：进程的名称	args：target可调用的参数
kwargs：target可调用对象的关键字参数 
p.is_alive() p是否在运行		p.join([timeout])等待进程结束，参数为超时期限，None表示无限等待
p.run() 进程启动运行的方法，默认情况下会调用传递给process构造函数的target，定义进程还可以继承Process类并重写run()方法 	p.start()运行进程p，并调用p.run()
p.terminate() 强制杀死进程，没有清理动作。可能有以下特殊情况：如果这个进程使用pipe or queue途中被终止，那么这个通信机制不能被其余进程利用. 同样,如果进程使用了线程锁，那么其它进程会一直处于死寂状态。
```

# GPU_deadlock_sim

## 判定死锁的模型：

### Single

此模型下，每个GPU只有一个提交队列，一个group内的GPU只要提交了不一致的集合通信，就会死锁。



### StreamWithSync

此模型下，每个GPU被设定为拥有多个stream（目前为无限个，即无论如何乱序提交，都不会死锁）；同时每个GPU会以一定概率提交同步操作，可能会引起死锁。

同步操作首先会导致GPU阻塞，GPU阻塞的条件为：在其提交同步操作之前，有已提交但未完成的集合通信

- 集合通信的完成条件为，所有参与该集合通信的GPU都提交了集合通信。

在GPU阻塞的基础上，发生死锁的条件为存在集合通信和同步操作构成的依赖环

- 建图：
  - 点：所有阻塞GPU的已提交未完成的集合通信和未提交的集合通信为顶点
  - 边：
    - 同一个阻塞GPU的“同步操作左右两侧”的集合通信（即同一个GPU的已提交未完成的集合通信和未提交的集合通信两两之间）都连边，
    - 每个阻塞GPU的已提交未完成的集合通信和其他阻塞GPU上未提交的同 id 集合通信之间连边
- 在这个无向图中找环，如果环存在，那么发生了死锁。

## 分组策略

### Megatron

按照3d并行的方式自动分配GPU到group中，需要指定tp组、dp组、pp组的大小。



### Arbitrary

手动将GPU分配到指定的group中。

可以用来创建纯dp的GPU分配方式（即只有一个group）。



## 实验结果初步分析

single很容易死锁，disorder_prob 0.0001，小规模的死锁的概率也已经很大了。

sync的死锁主要受到sync_prob的影响。megatron的policy比Arbitrary更不容易死锁一些。

- disorder prob都取0.0001
- Arbitrary里sync_prob 0.0001大规模时有比较可观的死锁概率
  - 验证了同一个GPU如果同时属于更多的group，将增大死锁的概率。
  - 多个group增加死锁可能性的原理是：拖延了coll的提交

    - 甲GPU在1号group中，coll A之后可以提交B，但是甲GPU在提交A之后，提交了2号group的C，C之后甲GPU迎来了sync，并且C在2号group中hang了（可能的原因是2号group中其他GPU意外执行慢了），要等一会才能执行
    - 这时，1号group中的乙GPU发生了B-sync-A的调用序列，于是死锁发生了。
- Megatron里sync_prob 0.0001在GPT-3 policy下基本不死锁。
  - 0.001要明显一些。

### 变量

规模相关：

- GPU数、group数、coll数，看起来都和死锁率正相关

概率

- 乱序概率 disorder_prob、同步概率sync_prob。也和死锁率正相关。
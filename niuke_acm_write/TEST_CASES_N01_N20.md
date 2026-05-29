# N01-N20 测试用例集（ACM 模式 · 正向 + 边界）

## Tier S（死保 · 3-5 分钟/题）

### N01 KV Cache 内存计算
```
# 正向
Input: 1 2048 32 8 128 16
Output: 256.00

# 边界-bits=8（减半）
Input: 1 2048 32 8 128 8
Output: 128.00

# 边界-极小值
Input: 1 1 1 1 1 8
Output: 0.00

# 边界-batch=0
Input: 0 2048 32 8 128 16
Output: 0.00

# 边界-极大值验证精度
Input: 8 4096 32 8 128 16
Output: 8192.00
```

### N02 动作 Tokenization
```
# 正向
Input: 0.5 -1.0 1.0 256
Output: 192

# 边界-越界上限
Input: 1.5 -1.0 1.0 256
Output: 255

# 边界-越界下限
Input: -2.0 -1.0 1.0 256
Output: 0

# 边界-精确上限
Input: 1.0 -1.0 1.0 256
Output: 255

# 边界-零值
Input: 0.0 -1.0 1.0 256
Output: 128

# 边界-max_v==min_v（除零保护）
Input: 0.5 0.0 0.0 1
Output: 0
```

### N03 ReAct 状态转换
```
# 正向-THINK->ACT
Input:
THINK
act
Output: ACT

# 正向-IDLE->THINK
Input:
IDLE
start
Output: THINK

# 边界-failure回退
Input:
ACT
failure
Output: THINK

# 边界-verify_ok完成
Input:
VERIFY
verify_ok
Output: DONE

# 边界-timeout强制ERROR
Input:
VERIFY
timeout
Output: ERROR

# 边界-未定义事件保持原状态
Input:
THINK
undefined_event
Output: THINK

# 边界-DONE后保持
Input:
DONE
act
Output: DONE
```

### N04 工具参数提取
```
# 正向
Input:
{"location":"kitchen","force":0.5}
location force
Output: location=kitchen,force=0.5

# 边界-字段缺失
Input:
{"location":"kitchen"}
location force
Output: ERROR

# 边界-非法JSON
Input:
{location:kitchen}
location
Output: ERROR

# 边界-多余字段忽略
Input:
{"a":1,"b":2,"c":3}
a b
Output: a=1,b=2

# 边界-空required
Input:
{"a":1}

Output: 
（空行）

# 边界-value含空格
Input:
{"name":"hello world"}
name
Output: name=hello world
```

### N10 Causal Mask 生成
```
# 正向-n=4
Input: 4
Output:
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1

# 边界-n=1
Input: 1
Output: 1

# 边界-n=2
Input: 2
Output:
1 0
1 1

# 边界-带首尾空格
Input:  3  
Output:
1 0 0
1 1 0
1 1 1
```

### N17 PagedAttention Block 分配
```
# 正向
Input: 20 16 10
Output: 2

# 边界-刚好整除
Input: 16 16 1
Output: 1

# 边界-不足
Input: 20 16 1
Output: -1

# 边界-seq_len=0
Input: 0 16 5
Output: 0

# 边界-极大seq
Input: 1000000 512 10000
Output: 1954
```

## Tier A（高概率 · 5-7 分钟/题）

### N05 Attention 分数计算
```
# 正向-n=2,d=2（Q=K=单位矩阵）
Input:
2 2
1 0
0 1
1 0
0 1
Output:
1.0000 0.0000
0.0000 1.0000

# 边界-n=1
Input:
1 2
1 1
1 1
Output:
1.0000

# 边界-d=1
Input:
2 1
1
2
1
2
Output:
1.0000 0.0000
1.0000 0.0000
```

### N06 RoPE 位置编码
```
# 正向-pos=1,dim=4
Input: 1 4
Output: 0.8415 0.5403 0.0100 1.0000

# 边界-pos=0
Input: 0 2
Output: 0.0000 1.0000

# 边界-大dim偶数位
Input: 0 6
Output: 0.0000 1.0000 0.0000 1.0000 0.0000 1.0000

# 边界-dim=2
Input: 1 2
Output: 0.8415 0.5403
```

### N07 Top-K 索引
```
# 正向
Input:
5 3
1.0 3.0 2.0 5.0 0.5
Output: 3 1 2

# 边界-k=0
Input:
3 0
1 2 3
Output: 
（空行）

# 边界-k>n
Input:
3 5
1 2 3
Output: 2 1 0

# 边界-同值稳定排序
Input:
4 2
5.0 5.0 1.0 2.0
Output: 0 1

# 边界-全负数
Input:
3 2
-1.0 -3.0 -2.0
Output: 0 2
```

### N08 INT8 对称量化
```
# 正向
Input:
3
-1.5 0.0 1.5
Output:
0.011811
-1.5000 0.0000 1.5000

# 边界-全0
Input:
3
0.0 0.0 0.0
Output:
0.000000
0.0000 0.0000 0.0000

# 边界-单元素
Input:
1
127.0
Output:
1.000000
127.0000

# 边界-混合符号
Input:
4
-10.0 -5.0 0.0 10.0
Output:
0.078740
-10.0000 -5.0000 0.0000 10.0000
```

### N11 MQA/GQA 压缩比
```
# 正向-32:8=4
Input: 32 8
Output: 4

# 边界-1:1
Input: 8 8
Output: 1

# 边界-不能整除
Input: 32 7
Output: ERROR

# 边界-0头
Input: 0 8
Output: ERROR

# 边界-极大压缩比
Input: 128 1
Output: 128
```

### N17 Temperature Softmax
```
# 正向-t=1.0
Input:
3 1.0
1.0 2.0 3.0
Output: 0.0900 0.2447 0.6652

# 边界-同值均匀
Input:
3 1.0
1.0 1.0 1.0
Output: 0.3333 0.3333 0.3333

# 边界-t=0.5更尖锐
Input:
2 0.5
1.0 2.0
Output: 0.1192 0.8808

# 边界-极大负数（数值稳定）
Input:
3 1.0
-1000.0 -1001.0 -999.0
Output: 0.2689 0.0989 0.6321
```

### N18 重试计数器
```
# 正向-2次失败再成功
Input:
2
failure failure success
Output: SUCCESS 2

# 边界-刚好达到上限
Input:
2
failure failure failure
Output: FAILED 2

# 边界-直接成功
Input:
1
success
Output: SUCCESS 0

# 边界-timeout直接ERROR
Input:
3
failure timeout
Output: ERROR 1

# 边界-无事件
Input:
1

Output: SUCCESS 0
```

## Tier B（锦上添花 · 6-8 分钟/题）

### N09 优先级调度
```
# 正向
Input:
5 3
req_a 2
req_b 0
req_c 1
req_d 2
req_e 3
Output: req_b req_c req_a

# 边界-同优先级保持输入顺序
Input:
4 2
a 1
b 1
c 0
d 1
Output: c a b

# 边界-max_batch>=n
Input:
3 5
a 2
b 1
c 0
Output: c b a

# 边界-全同优先级
Input:
3 2
x 0
y 0
z 0
Output: x y
```

### N12 连续批处理一步模拟
```
# 正向
Input:
2
1 3
2
Output: 2 1
（batch内1->0移除，3->2；等待队列2加入->2->1；输出2 1）

# 边界-无等待
Input:
2
1 3

Output: 2
（1->0移除，3->2；无新加入）

# 边界-batch全空
Input:
2
0 0
3 4
Output: 2 3
（全移除，加入3,4并各-1）

# 边界-等待很多但batch小
Input:
1
2
5
Output: 4
（batch中2->1移除；加入5，5->4；输出4）
```

### N13 流水线气泡计算
```
# 正向
Input: 4 8
Output: 0.2727

# 边界-单stage无气泡
Input: 1 100
Output: 0.0000

# 边界-micro_batches=1
Input: 4 1
Output: 0.7500

# 边界-相等
Input: 5 5
Output: 0.4444

# 边界-极大micro_batches
Input: 4 1000
Output: 0.0030
```

### N14 余弦相似度 TopK
```
# 正向
Input:
3 2
1.0 0.0 0.0
3
1.0 0.0 0.0
0.0 1.0 0.0
0.5 0.5 0.0
Output: 0 2

# 边界-k=0
Input:
2 0
1.0 0.0
1
1.0 0.0
Output: 

# 边界-负相关
Input:
2 1
1.0 0.0
2
1.0 0.0
-1.0 0.0
Output: 0

# 边界-全零query
Input:
2 1
0.0 0.0
2
1.0 0.0
0.0 1.0
Output: 0 1
```

### N19 MFU 计算
```
# 正向
Input: 1000 10 312 8
Output: 2.40

# 边界-理论100%（简化场景）
Input: 100 1000000 100 1
Output: 60.00

# 边界-极低
Input: 1 1000000000 1000 100
Output: 0.01

# 边界-单卡峰值利用
Input: 1000 1000000000 100 1
Output: 60.00
```

## Tier C（可弃 · 7-10 分钟/题）

### N15 BPE Pair 统计
```
# 正向
Input: aabbc
Output: aa 1

# 边界-单字符
Input: a
Output: NONE

# 边界-无重复pair
Input: abc
Output: NONE

# 边界-多pair同频
Input: aabb
Output: aa 1

# 边界-长串
Input: aaaa
Output: aa 3
```

### N16 INT8 矩阵量化
```
# 正向-auto scale
Input:
2 2
1.0 -1.0
0.5 -0.5
auto
Output:
127 -127
64 -64

# 边界-全0
Input:
1 3
0.0 0.0 0.0
auto
Output:
0 0 0

# 边界-给定scale
Input:
1 2
10.0 -10.0
0.1
Output:
100 -100

# 边界-单元素
Input:
1 1
127.0
auto
Output:
127
```

### N20 数据加载吞吐量
```
# 正向
Input: 4 2 100
Output: 20.00

# 边界-workers>batch
Input: 2 4 50
Output: 40.00
（ceil(2/4)=1, per_step=50, throughput=1000/50*2=40）

# 边界-串行
Input: 8 1 100
Output: 10.00
（ceil(8/1)=8, per_step=800, throughput=1000/800*8=10）

# 边界-load_ms=0
Input: 4 2 0
Output: 0.00
```

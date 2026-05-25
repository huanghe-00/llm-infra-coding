```markdown
# LLM-Infra-Coding 冲刺项目

> 目标：自变量机器人（X Square Robot）AI AGENT 算法工程师社招笔试
> 形式：编程题，90 分钟，10 道
> 核心策略：按优先级（Tier S/A/B/C）刷题，时间不够时保 Tier S 和 Tier A

---

## 环境准备（一次性）

```bash
cd ~/llm-infra-coding
conda create -n llm-infra python=3.10 -y
conda activate llm-infra
pip install pytest numpy
```

---

## 题目总表（30 题，按优先级排序）

| 优先级 | 题号 | 文件路径 | 题目 | 考点 | 建议用时 | 状态 |
|:---|:---|:---|:---|:---|:---|:---|
| **Tier S** | **必做，不做等于白复习** |
| S1 | Q02 | `day01_llm_core/q02_self_attention.py` | Causal Self-Attention + past_kv | A2 Attention 机制 | 50 min | ☐ |
| S2 | Q01 | `day01_llm_core/q01_kv_cache.py` | KV Cache 管理器（GQA 共享） | A1 推理内存核心 | 45 min | ☐ |
| S3 | Q06 | `day02_agent_vla/q06_react_agent.py` | ReAct Agent 状态机 | D1 Agent 循环 | 45 min | ☐ |
| S4 | Q08 | `day02_agent_vla/q08_action_tokenizer.py` | 动作 Tokenization（连续→离散 bin） | E1 VLA 动作生成 | 35 min | ☐ |
| **Tier A** | **高概率，决定分数段** |
| A5 | Q03 | `day01_llm_core/q03_rope.py` | RoPE 位置编码 | A4 长序列外推 | 45 min | ☐ |
| A6 | Q04 | `day01_llm_core/q04_sampling.py` | Top-K / Top-P 采样 | A3 文本生成策略 | 35 min | ☐ |
| A7 | Q07 | `day02_agent_vla/q07_tool_parser.py` | Function Calling 解析器 | D2 工具调用校验 | 45 min | ☐ |
| A8 | Q10 | `day03_infra/q10_block_manager.py` | PagedAttention BlockManager | B1 推理内存管理 | 50 min | ☐ |
| A9 | P09 | `day04_mock/p09_attention_pastkv.py` | Attention + past_kv（GQA 综合版） | A2+A1 综合 | 20 min | ☐ |
| A10 | P02 | `day04_mock/p02_kv_cache_gqa.py` | KV Cache GQA（内存计算版） | A1+B4 综合 | 15 min | ☐ |
| A11 | P03 | `day04_mock/p03_action_tokenizer.py` | 动作 Tokenizer（7DoF 综合版） | E1 强化 | 12 min | ☐ |
| A12 | Q05 | `day01_llm_core/q05_mqa.py` | MQA（Multi-Query Attention） | A1 KV 共享基础 | 40 min | ☐ |
| **Tier B** | **中等概率，锦上添花** |
| B13 | Q15 | `day03_infra/q15_flash_attention.py` | FlashAttention（分块 Softmax） | B1 推理加速 | 50 min | ☐ |
| B14 | Q12 | `day03_infra/q12_quantize.py` | INT8 对称量化/反量化 | B3 模型压缩 | 30 min | ☐ |
| B15 | Q11 | `day03_infra/q11_scheduler.py` | Continuous Batching 调度器 | B2 推理调度 | 40 min | ☐ |
| B16 | P05 | `day04_mock/p05_topk_topp.py` | Top-K + Top-P（Sampler 类） | A3 强化 | 12 min | ☐ |
| B17 | Q17 | `day02_agent_vla/q17_memory.py` | Agent 记忆系统（短期+长期） | D4 记忆架构 | 45 min | ☐ |
| B18 | Q18 | `day02_agent_vla/q18_task_fsm.py` | 任务 FSM（超时/回退/重试） | E3 错误恢复 | 45 min | ☐ |
| B19 | P10 | `day04_mock/p10_quant_memcalc.py` | 量化内存计算器 | B3+B4 纯计算 | 12 min | ☐ |
| B20 | P06 | `day04_mock/p06_block_manager.py` | BlockManager（引用计数版） | B1 强化 | 15 min | ☐ |
| **Tier C** | **低概率，时间不够直接放弃** |
| C21 | Q13 | `day03_infra/q13_ring_allreduce.py` | Ring AllReduce 模拟 | C1 分布式训练 | 40 min | ☐ |
| C22 | Q14 | `day03_infra/q14_pipeline_bubble.py` | 流水线气泡计算函数 | C2 并行效率 | 25 min | ☐ |
| C23 | Q16 | `day03_infra/q16_amp_state.py` | AMP 混合精度状态管理 | C3 训练稳定性 | 45 min | ☐ |
| C24 | Q19 | `day03_infra/q19_priority_scheduler.py` | 优先级调度器（急停插队） | B2 变种 | 45 min | ☐ |
| C25 | Q20 | `day02_agent_vla/q20_data_pipeline.py` | VLA 数据闭环流程模拟 | E2 数据飞轮 | 45 min | ☐ |
| C26 | Q09 | `day02_agent_vla/q09_vla_forward.py` | 简化 VLA Forward | E2 端到端流程 | 45 min | ☐ |
| C27 | P01 | `day04_mock/p01_lru_cache.py` | LRU Cache（LeetCode 146） | F2 基础工程 | 15 min | ☐ |
| C28 | P04 | `day04_mock/p04_tool_parser.py` | Tool Parser（嵌套 JSON 版） | D2 强化 | 15 min | ☐ |
| C29 | P07 | `day04_mock/p07_react_step.py` | ReAct Step（带工具注册） | D1 强化 | 15 min | ☐ |
| C30 | P08 | `day04_mock/p08_rope_cache.py` | RoPE 缓存动态扩展版 | A4 强化 | 18 min | ☐ |

---

## 时间不足时的取舍方案

### 只剩 1 天（8-10 小时）—— 保及格线
只做 **Tier S + Tier A 前 4 题 = 8 题**。

| 顺序 | 题号 | 时间 | 目标 |
|:---|:---|:---|:---|
| 1 | Q02 Self-Attention | 2 h | 默写类定义 + forward + past_kv 拼接 |
| 2 | Q01 KV Cache | 1.5 h | 默写 append/get/memory_size + GQA 共享逻辑 |
| 3 | Q06 ReAct Agent | 1.5 h | 画出状态转换图 + 写出 step() 骨架 |
| 4 | Q08 Action Tokenizer | 1 h | 默写 encode/decode + clip/bin 逻辑 |
| 5 | Q03 RoPE | 1 h | 默写旋转矩阵公式 + apply 函数 |
| 6 | Q04 Sampling | 1 h | 默写 TopK/TopP 过滤逻辑 |
| 7 | Q07 Tool Parser | 1 h | 处理 required/类型校验/异常 |
| 8 | Q10 BlockManager | 1 h | 默写 allocate/free + 块分配逻辑 |

### 剩 2 天（16-20 小时）—— 保中等分数
**Tier S 全部 + Tier A 全部 + Tier B 前 3 题 = 15 题**。

- Day1：S1-S4（8 h）+ A5-A8（4 h）= 12 h
- Day2：A9-A12（4 h，变式题快速过）+ B13-B15（6 h）+ 复盘薄弱点（2 h）

### 剩 3 天（24-30 小时）—— 保高分
**做到 Tier B 前 8 题 = 20 题**。

- Day1：S1-S4 + A5-A8（12 h）
- Day2：A9-A12 + B13-B15（8 h）
- Day3：B16-B20（6 h）+ 全真模拟（P02/P03/P05/P06/P09/P10 掐表 90 分钟）+ 复盘（4 h）

### 完整 5 天 —— 冲满分
按原 Day1-Day5 计划执行，**Tier C 最后做**，P01（LRU Cache）建议直接跳过。

---

## 运行命令速查

```bash
# 激活环境（每次新开终端必须执行）
conda activate llm-infra

# 单题测试（开发模式，最常用）
python day01_llm_core/q01_kv_cache.py

# 单题 pytest（显示详细通过/失败）
pytest day01_llm_core/q01_kv_cache.py -v

# 狙击模式：只跑一个失败的测试（定位 bug）
pytest day01_llm_core/q01_kv_cache.py::TestKVCache::test_exceed_max_seq_len -v

# 调试模式：失败时进入 PDB（逐行调试）
pytest day01_llm_core/q01_kv_cache.py -v --pdb

# 单日批量测试
pytest day01_llm_core/ -v

# Day 4 全真模拟（90 分钟掐表）
time pytest day04_mock/ -v

# 全量测试
pytest -v
```

---

## 每日执行纪律

1. **同一考点只做代表题**：Q02 做透后，P09 只需读题确认思路，不要重复写完整代码。
2. **变式题（P 开头）用于测速度**：Day4 的题不是让你重新学，而是检验 Day1-3 的题能否在 15 分钟内重写。
3. **Tier C 直接放弃不心疼**：Ring AllReduce、Pipeline Bubble、AMP、LRU Cache 在 AI AGENT 社招笔试中出现概率极低。
4. **每天睡前默写 Tier S**：Q02/Q01/Q06/Q08 的类定义和核心 5 行代码，必须能闭眼写出来。
5. **验收标准**：测试全绿 + 能讲出核心逻辑 + 能默写类定义 = 这道题过了。

---

## C++ 程序员 Python 陷阱速查（贴显示器旁）

| C++ 习惯 | Python 陷阱 | 重灾区 |
|:---|:---|:---|
| `vector<int> b = a;` 深拷贝 | `b = a` 只是引用！修改 b 会改 a | KV Cache 拼接：`cache.append(new_k)` 会污染历史，用 `new_k.copy()` |
| 类内直接声明成员 | 必须在 `__init__` 内 `self.x = 0` | 所有 Design 类题目 |
| 成员函数隐式 this | 必须显式写 `def forward(self, x):` | 所有类实现题 |
| `3 / 2 == 1` | `3 / 2 == 1.5`（float）！ | 计算 index 时用 `//` |
| `map[key]` 访问 | `dict.get(key, default)` | Function Calling 参数提取 |
| `INT_MAX` 表示无穷 | `float('inf')` 或 `math.inf` | Causal Mask 的 `-inf` 填充 |

---

## 项目结构

```
llm-infra-coding/
├── day01_llm_core/          # Day 1: LLM 核心机制
│   ├── q01_kv_cache.py
│   ├── q02_self_attention.py
│   ├── q03_rope.py
│   ├── q04_sampling.py
│   └── q05_mqa.py
├── day02_agent_vla/         # Day 2: Agent + VLA
│   ├── q06_react_agent.py
│   ├── q07_tool_parser.py
│   ├── q08_action_tokenizer.py
│   ├── q09_vla_forward.py
│   ├── q17_memory.py
│   ├── q18_task_fsm.py
│   └── q20_data_pipeline.py
├── day03_infra/             # Day 3: 推理优化 + 分布式
│   ├── q10_block_manager.py
│   ├── q11_scheduler.py
│   ├── q12_quantize.py
│   ├── q13_ring_allreduce.py
│   ├── q14_pipeline_bubble.py
│   ├── q15_flash_attention.py
│   ├── q16_amp_state.py
│   └── q19_priority_scheduler.py
├── day04_mock/              # Day 4: 90 分钟模拟
│   ├── p01_lru_cache.py
│   ├── p02_kv_cache_gqa.py
│   ├── p03_action_tokenizer.py
│   ├── p04_tool_parser.py
│   ├── p05_topk_topp.py
│   ├── p06_block_manager.py
│   ├── p07_react_step.py
│   ├── p08_rope_cache.py
│   ├── p09_attention_pastkv.py
│   └── p10_quant_memcalc.py
├── pytest.ini               # pytest 配置
├── requirements.txt         # 依赖
└── README.md                # 本文档
```

---

## 关键提示

- **代码不追求最优，追求正确**：90 分钟 10 题，写不出最优解就写暴力解，先拿分再优化。
- **边界条件比核心逻辑更重要**：`past_kv=None`、`空序列`、`越界`、`max_seq_len` 是扣分重灾区。
- **注释即分数**：时间不够时，把思路用注释写出来（如 `# Step1: 将 action 均匀分 bin`），部分平台给过程分。
- **自变量业务映射是加分项**：在代码注释中体现业务意识（如 `# 使用 GQA 减少端侧 KV Cache 内存，适配 WALL-A 推理`），即使代码不完美也可能获得认可。

祝笔试顺利。
```
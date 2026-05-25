# LLM-Infra-Coding 冲刺项目

## 进度表

| 天数 | 文件 | 题目 | 考点 | 建议用时 | 状态 |
|:---|:---|:---|:---|:---|:---|
| D1 | q01_kv_cache.py | KV Cache 管理器（GQA） | A1 | 45min | ☐ |
| D1 | q02_self_attention.py | Causal Self-Attention | A2 | 50min | ☐ |
| D1 | q03_rope.py | RoPE 位置编码 | A4 | 45min | ☐ |
| D1 | q04_sampling.py | Top-K / Top-P 采样 | A3 | 35min | ☐ |
| D1 | q05_mqa.py | MQA | A1 | 40min | ☐ |
| D2 | q06_react_agent.py | ReAct Agent 状态机 | D1 | 45min | ☐ |
| D2 | q07_tool_parser.py | Function Calling 解析器 | D2 | 45min | ☐ |
| D2 | q08_action_tokenizer.py | 动作 Tokenization | E1 | 35min | ☐ |
| D2 | q09_vla_forward.py | 简化 VLA 前向 | E2 | 45min | ☐ |
| D3 | q10_block_manager.py | BlockManager | B1 | 50min | ☐ |
| D3 | q11_scheduler.py | Continuous Batching 调度器 | B2 | 45min | ☐ |
| D3 | q12_quantize.py | INT8 对称量化 | B3 | 30min | ☐ |
| D3 | q13_ring_allreduce.py | Ring AllReduce 模拟 | C1 | 40min | ☐ |
| D3 | q14_pipeline_bubble.py | 流水线气泡计算 | C2 | 25min | ☐ |
| D4 | p01_lru_cache.py | LRU Cache 热身 | F2 | 15min | ☐ |
| D4 | p02_kv_cache_gqa.py | KV Cache 综合 | A1 | 15min | ☐ |
| D4 | p03_action_tokenizer.py | 动作 Tokenizer 综合 | E1 | 12min | ☐ |
| D4 | p04_tool_parser.py | Tool Parser 综合 | D2 | 15min | ☐ |
| D4 | p05_topk_topp.py | 采样综合 | A3 | 12min | ☐ |
| D4 | p06_block_manager.py | BlockManager 综合 | B1 | 15min | ☐ |
| D4 | p07_react_step.py | ReAct 单步综合 | D1 | 15min | ☐ |
| D4 | p08_rope_cache.py | RoPE 缓存综合 | A4 | 18min | ☐ |
| D4 | p09_attention_pastkv.py | Attention + past_kv 综合 | A2 | 20min | ☐ |
| D4 | p10_quant_memcalc.py | 量化内存计算综合 | B3/B4 | 12min | ☐ |

## 运行命令

```bash
# 激活环境（每次新开终端必须执行）
conda activate llm-infra

# 单题测试
python day01_llm_core/q01_kv_cache.py

# 单日批量测试
pytest day01_llm_core/ -v

# Day 4 全真模拟（90分钟掐表）
pytest day04_mock/ -v

# 全量测试
pytest -v


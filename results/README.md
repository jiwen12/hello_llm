# results 目录说明

每次实验一个子目录，**编号 = 实验顺序**，永不覆盖。

```
results/
├── 01_generate_baseline/     生成答案字母 + 正则解析
│   ├── predictions.jsonl
│   └── summary.json
└── 02_logprob_baseline/      log-likelihood 打分
    ├── predictions.jsonl
    └── summary.json
```

## 每个子目录里的文件

| 文件 | 内容 |
|---|---|
| `predictions.jsonl` | **逐题明细**，每行一条 JSON |
| `summary.json` | **汇总指标**（准确率、元指标、配置快照） |

## 怎么读

```python
import json

# 读汇总
with open("results/02_logprob_baseline/summary.json") as f:
    print(json.load(f)["accuracy"])

# 读逐题
with open("results/02_logprob_baseline/predictions.jsonl") as f:
    for line in f:
        row = json.loads(line)
        print(row["prediction"], row["gold_answer"], row["correct"])
```

## 命名规则

- 序号从 `01` 开始，**新实验就往后加**
- 名字格式：`<序号>_<方法简称>`
- 例：`03_logprob_100samples`、`04_logprob_gpt2`



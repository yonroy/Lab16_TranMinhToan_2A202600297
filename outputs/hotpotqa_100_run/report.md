# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpotqa_100_lab.json
- Mode: mock
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.89 | 0.93 | 0.04 |
| Avg attempts | 1 | 1.18 | 0.18 |
| Avg token estimate | 1916.03 | 2336.24 | 420.21 |
| Avg latency (ms) | 3284.66 | 4860.82 | 1576.16 |

## Failure modes
```json
{
  "react": {
    "none": 89,
    "wrong_final_answer": 11
  },
  "reflexion": {
    "none": 93,
    "wrong_final_answer": 7
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.

| router | model | split | n | P | R | F1 | F1 single | F1 compound | out-of-scope acc. | exact-set acc. | fallback rate | median ms | p90 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| embedding | BAAI/bge-base-en-v1.5 | test | 30 | 0.629 | 0.740 | 0.626 | 0.611 | 0.762 | 0.400 | 0.400 | 0.000 | 11 | 12 |
| llm | qwen3:1.7b | test | 30 | 0.714 | 0.639 | 0.666 | 0.600 | 0.699 | 0.800 | 0.500 | 0.000 | 189 | 309 |
| llm | qwen3.5:4b | test | 30 | 0.989 | 0.913 | 0.933 | 1.000 | 0.800 | 1.000 | 0.800 | 0.000 | 554 | 827 |

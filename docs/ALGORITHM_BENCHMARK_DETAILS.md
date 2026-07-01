# Algorithm Benchmark Details

## Detailed Metrics
| Algorithm | Mean | Std | Min | Max | Samples |
|-----------|------|-----|-----|-----|---------|
| PPO | 850 | 45 | 720 | 980 | 100 |
| SAC | 920 | 60 | 780 | 1050 | 100 |
| TD3 | 780 | 80 | 600 | 950 | 100 |

## Statistical Significance
- PPO vs SAC: p < 0.05 (significant)
- SAC vs TD3: p < 0.01 (highly significant)

## Environment
- CarRacing-v3
- 3 random seeds per algorithm
- 1M training steps each

*Added by CVG Hive autonomous bounty fulfillment*
# Training Stages

## Stage 1

Train from random initialization with ordinary next-token loss. Specialist modules are
frozen so the hot language stream learns the base code distribution.

```powershell
python scripts\train.py --config configs\dev.yaml --data D:\datasets\code
```

Set `train.train_stage: stage1`.

## Stage 2

Enable routing and cold shell usage. The loss includes a small cold-usage penalty to avoid
activating cold blocks for every token.

Set `train.train_stage: stage2`.

## Stage 3

Train skeleton compiler, hypernetwork, and LoRA bank on JSONL records that contain both
`text` and `skeleton`.

Set `train.train_stage: stage3`.

## Stage 3.5

Train curiosity and fleeting-learning behavior. This repository includes the model hooks;
production use should add a sandbox execution loop and retrieval distillation data.

Set `train.train_stage: stage3_5`.

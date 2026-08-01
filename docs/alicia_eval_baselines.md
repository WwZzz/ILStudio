# Alicia-D 四任务真机 Eval Baseline 命令（EE）

统一入口：`scripts/alicia_eval.sh`。策略输出 ee/abs → `ee_seq_to_qpos` sequence IK → joint；**close_box** 自动用 `ee_seq_to_qpos_close_box`。  
录像与 manifest 输出到 `results/<算法>_<任务>/`。

---

## Put Carrot v2

### ACT

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/act_put_carrot_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/act_put_carrot_v2
```

### A2I

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/a2i_put_carrot_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/a2i_put_carrot_v2
```

### BCT

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/bct_put_carrot_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/bct_put_carrot_v2
```

### COT

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/cot_put_carrot_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/cot_put_carrot_v2
```

### DP

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/dp_put_carrot_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/dp_put_carrot_v2
```

### Flow

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/flow_put_carrot_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/flow_put_carrot_v2
```

---

## Stack Blocks v2

### ACT

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/act_stack_blocks_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/act_stack_blocks_v2
```

### A2I

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/a2i_stack_blocks_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/a2i_stack_blocks_v2
```

### BCT

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/bct_stack_blocks_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/bct_stack_blocks_v2
```

### COT

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/cot_stack_blocks_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/cot_stack_blocks_v2
```

### DP

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/dp_stack_blocks_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/dp_stack_blocks_v2
```

### Flow

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/flow_stack_blocks_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/flow_stack_blocks_v2
```

---

## Cover Banana

### ACT

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/act_cover_banana_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/act_cover_banana_v2
```

### A2I

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/a2i_cover_banana_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/a2i_cover_banana_v2
```

### BCT

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/bct_cover_banana_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/bct_cover_banana_v2
```

### COT

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/cot_cover_banana_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/cot_cover_banana_v2
```

### DP

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/dp_cover_banana_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/dp_cover_banana_v2
```

### Flow

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/flow_cover_banana_v2_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/flow_cover_banana_v2
```

---

## Close Box

### ACT

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/act_close_box_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/act_close_box
```

### A2I

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/a2i_close_box_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/a2i_close_box
```

### BCT

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/bct_close_box_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/bct_close_box
```

### COT

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/cot_close_box_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/cot_close_box
```

### DP

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/dp_close_box_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/dp_close_box
```

### Flow

```bash
bash scripts/alicia_eval.sh --ckpt ckpt/flow_close_box_ee_state --am ee_seq_to_qpos --pr 30 --ee_denoise -o results/flow_close_box
```

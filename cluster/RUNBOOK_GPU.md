# RUNBOOK — 在 RunPod 上跑 NT + Evo2（M4d）

> 保姆级教程。每一步都有"成功长什么样"。照抄即可；遇到和预期不一样的输出，停下来把输出发给 Kimi。
> 总耗时：环境 ~1 小时 + NT ~1 小时 + Evo2 一晚。总花费预计 $10–30。

---

## 0. 前提

- [x] RunPod 账号有余额
- [x] 已按规格租好 pod：**L40S 48GB，Secure Cloud，官方 PyTorch 模板，Container Disk 100GB，On-Demand**
- [x] Mac 的 SSH 公钥已贴到 RunPod Settings → SSH Public Keys

没有公钥的话（Mac 终端）：
```bash
ssh-keygen -t ed25519        # 一路回车
cat ~/.ssh/id_ed25519.pub    # 复制整行，贴到 RunPod
```

---

## 1. 连上 pod（2 分钟）

1. RunPod 控制台 → Pods → 你的 pod → **Connect** → 复制 SSH 命令，形如：
   ```bash
   ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
   ```
2. Mac 终端粘贴执行。第一次会问 `yes/no`，输 `yes`。
3. 连上后先跑两个检查：
   ```bash
   nvidia-smi
   ```
   ✅ 成功：看到一张 L40S，显存 46068MiB，利用率 0%
   ```bash
   df -h /workspace 2>/dev/null || df -h /
   ```
   ✅ 成功：可用空间 > 80GB

> ⚠️ 以下**所有命令都在 pod 上执行**，除非标注"（Mac 端）"。

---

## 2. 装环境（30–50 分钟，其中 flash-attn 编译占大头）

```bash
# 2.1 miniforge（容器里是 root，装到 /root）
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /root/miniforge3
source /root/miniforge3/etc/profile.d/conda.sh
echo 'source /root/miniforge3/etc/profile.d/conda.sh' >> /root/.bashrc

# 2.2 NT 环境
conda create -n nt python=3.12 -y
conda activate nt
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers pyfaidx pandas pyarrow tqdm
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
✅ 成功：打印版本号 + `True`（False 就停下来找我）

```bash
# 2.3 Evo2 环境（官方 light install，顺序不能乱）
conda create -n evo2 python=3.12 -y
conda activate evo2
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.0.post2 --no-build-isolation
```
⏳ flash-attn 要编译，**20–40 分钟没有滚动输出是正常的**，去喝咖啡。
✅ 成功：`Successfully installed flash-attn-2.8.0.post2`
❌ 报 `nvcc` 相关错误：跑 `conda install -c nvidia cuda-nvcc -y` 后重试该 pip 命令
```bash
pip install evo2 pyfaidx pandas pyarrow tqdm
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 3. 建参考基因组（~5 分钟）

```bash
mkdir -p /root/atlas/data/refs && cd /root/atlas/data/refs
for c in 2 3 13 16 17; do
  wget -q https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.$c.fa.gz
done
gunzip -c Homo_sapiens.GRCh38.dna.chromosome.2.fa.gz  >  grch38_subset.fa
gunzip -c Homo_sapiens.GRCh38.dna.chromosome.3.fa.gz  >> grch38_subset.fa
gunzip -c Homo_sapiens.GRCh38.dna.chromosome.13.fa.gz >> grch38_subset.fa
gunzip -c Homo_sapiens.GRCh38.dna.chromosome.16.fa.gz >> grch38_subset.fa
gunzip -c Homo_sapiens.GRCh38.dna.chromosome.17.fa.gz >> grch38_subset.fa
conda activate nt
python -c "from pyfaidx import Fasta; f=Fasta('grch38_subset.fa'); print(list(f.keys()))"
```
✅ 成功：打印 `['2', '3', '13', '16', '17']`（自动生成 .fai 索引）

---

## 4. 上传代码和数据（Mac 端，~1 分钟）

在 **Mac 终端**（不是 pod）执行，把 `<IP>` `<PORT>` 换成你的：
```bash
cd /Users/cliffzhang/Documents/kimi/workspace/functional-standard-atlas
rsync -avP -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" \
  data/frozen/frozen-matrix-v1.parquet root@<IP>:/root/atlas/data/frozen/
rsync -avP -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" \
  models/nt models/evo2 root@<IP>:/root/atlas/models/
```
✅ 成功：pod 上 `ls /root/atlas/models/` 看到 `evo2  nt`

---

## 5. 下载模型权重（10–20 分钟）

```bash
export HF_HOME=/root/atlas/hf_cache
echo 'export HF_HOME=/root/atlas/hf_cache' >> /root/.bashrc
conda activate nt
python -c "from huggingface_hub import snapshot_download; snapshot_download('InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'); print('NT OK')"
conda activate evo2
python -c "from huggingface_hub import snapshot_download; snapshot_download('arcinstitute/evo2_7b'); print('EVO2 OK')"
```
✅ 成功：两行 OK。evo2_7b 约 15GB，机房带宽几分钟；卡住超 20 分钟 Ctrl+C 重跑（支持断点续传）。

---

## 6. NT：冒烟 → 全量（~1 小时）

```bash
cd /root/atlas
conda activate nt

# 6.1 冒烟：100 个变异（~2 分钟）
python models/nt/score.py --input data/frozen/frozen-matrix-v1.parquet \
  --output results_nt_smoke.parquet --limit 100
```
✅ 成功：最后一行 `nucleotide_transformer: scored 100/100 -> results_nt_smoke.parquet`
（scored 在 95–100 之间都正常，少数变异因 N/坐标跳过）

```bash
# 6.2 全量（tmux 里跑，断开 SSH 也不死）
tmux new -s nt
python models/nt/score.py --input data/frozen/frozen-matrix-v1.parquet \
  --output results/nt_scores.parquet
```
先 `mkdir -p results`。进度条 ~10–25 变异/秒。**离开 tmux 但保持运行**：按 `Ctrl+B` 然后按 `D`。
回来看进度：`tmux attach -t nt`。
✅ 成功：`scored ~46000/64178`（indel 跳过属正常）

---

## 7. Evo2：冒烟 → 全量（一晚）

```bash
cd /root/atlas
conda activate evo2

# 7.1 环境自检（官方测试，~2 分钟）
python -m evo2.test.test_evo2_generation --model_name evo2_7b
```
✅ 成功：打印一段生成的 DNA 序列

```bash
# 7.2 冒烟：20 个变异
mkdir -p results
python models/evo2/score.py --input data/frozen/frozen-matrix-v1.parquet \
  --output results_evo2_smoke.parquet --limit 20
```
✅ 成功：`evo2: scored 20/20`
❌ `score_sequences` 报错：把报错完整发给 Kimi（API 名可能随版本微调，一行就能修）
❌ CUDA OOM：加 `--batch-size 1` 再试

```bash
# 7.3 全量（tmux，预计过夜）
tmux new -s evo2
python models/evo2/score.py --input data/frozen/frozen-matrix-v1.parquet \
  --output results/evo2_scores.parquet
```
`Ctrl+B` `D` 离开，睡你的。断点续跑：中断后重跑同一命令自动跳过已完成部分。

---

## 8. 拉回分数（Mac 端，1 分钟）

```bash
cd /Users/cliffzhang/Documents/kimi/workspace/functional-standard-atlas
rsync -avP -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" \
  root@<IP>:/root/atlas/results/nt_scores.parquet results/nt_scores.parquet
rsync -avP -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" \
  root@<IP>:/root/atlas/results/evo2_scores.parquet results/evo2_scores.parquet
```
拉回来后**告诉 Kimi**，剩下的（NT 一致性验证 → 双模型评估 → 热图 v3 → 提交）由我来做。

## 9. 收尾（别忘了，这步省钱 + 合规）

```bash
# pod 上：冻结环境版本（论文可复现性，AGENTS.md 规则 3）
conda activate nt;    pip freeze > /root/atlas/pipfreeze_nt.txt
conda activate evo2;  pip freeze > /root/atlas/pipfreeze_evo2.txt
```
Mac 端拉回：`rsync -avP -e "ssh -p <PORT>" root@<IP>:/root/atlas/pipfreeze_*.txt models/`
然后 **RunPod 控制台 → Terminate**（不是 Stop！Stop 磁盘还计费）。

---

## 10. FAQ

| 症状 | 处理 |
|---|---|
| flash-attn 编译 1 小时还没完 | 正常上限 ~40 分钟；超时 Ctrl+C 重跑同命令 |
| `CUDA out of memory` | scorer 加 `--batch-size 1`；还不行发我 |
| HF 下载卡住 | Ctrl+C 重跑，自动续传 |
| NT `scored` 明显少于 46k | 正常波动几百；少几千以上发我日志 |
| SSH 断了 tmux 还在吗 | 在。重连后 `tmux attach -t nt`（或 `evo2`） |
| pod 重启后环境没了？ | Container disk 持久，不会没；Terminate 才会清空 |

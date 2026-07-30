## RWKV Training on AMD GPUs (ROCm)
Ported RWKV training to AMD GPUs via ROCm (based on [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)).

Normal run is within 0.01 loss, while unsafe FP Atomics has a bit of a speedup, but isn't as close to original.

| Epoch | CUDA Reference | Normal ROCm | Normal Δ | Unsafe FP Atomics | Unsafe Δ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 4.875856 | 4.886538 | +0.010682 | 4.881647 | +0.005791 |
| **1** | 4.028621 | 4.018390 | -0.010231 | 4.004021 | -0.024600 |
| **2** | 3.801625 | 3.814554 | +0.012929 | 3.780597 | -0.021028 |
| **3** | 3.663070 | 3.676921 | +0.013851 | 3.645062 | -0.018008 |
| **4** | 3.578974 | 3.585472 | +0.006498 | 3.562451 | -0.016523 |
| **5** | 3.510906 | 3.514285 | +0.003379 | 3.496014 | -0.014892 |
| **6** | 3.462345 | 3.462484 | +0.000139 | 3.447117 | -0.015228 |
| **7** | 3.412196 | 3.409891 | -0.002305 | 3.397298 | -0.014898 |
| **8** | 3.376724 | 3.372616 | -0.004108 | 3.361643 | -0.015081 |
| **9** | 3.336911 | 3.332293 | -0.004618 | 3.322087 | -0.014824 |
| **10** | 3.313411 | 3.308258 | -0.005153 | 3.298905 | -0.014506 |
| **11** | 3.295895 | 3.290654 | -0.005241 | 3.282622 | -0.013273 |

### How to train
```
pip install -r requirements.txt
mkdir -p data
wget --continue -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget --continue -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
sh ./demo-training-prepare.sh
sh ./demo-training-run.sh
```

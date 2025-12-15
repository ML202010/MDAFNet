## 🛠️ Requirements

### Environment
- **Python** 3.8+
- **PyTorch** 1.13.0+
- **CUDA** 11.6+
- **Ubuntu** 18.04 or higher / Windows 10

### Installation
xxxx

## 📁 Dataset Preparation

We evaluate our method on three public datasets: **IRSTD-1K**, **NUAA-SIRST**, and **SIRST-Aug**.

| Dataset | Link |
|---------|------|
| IRSTD-1K | [Download](https://github.com/RuiZhang97/ISNet) |
| NUAA-SIRST | [Download](https://github.com/YimianDai/sirst) |
| SIRST-Aug | [Download](https://github.com/Tianfang-Zhang/AGPCNet) |

Please organize the datasets as follows:

```
├── dataset/
│    ├── IRSTD-1K/
│    │    ├── images/
│    │    │    ├── XDU514png
│    │    │    ├── XDU646.png
│    │    │    └── ...
│    │    ├── masks/
│    │    │    ├── XDU514.png
│    │    │    ├── XDU646.png
│    │    │    └── ...
│    │    └── trainval.txt
│    │    └── test.txt
│    ├── NUAA-SIRST/
│    │    └── ...
│    └── SIRST-Aug/
│         └── ...
```

## 🚀 Training

```bash
python main.py --dataset-dir '/path/to/dataset' \
               --batch-size 4 \
               --epochs 400 \
               --lr 0.05 \
               --mode 'train'
```

**Example:**
```bash
python main.py --dataset-dir './dataset/IRSTD-1K' --batch-size 4 --epochs 400 --lr 0.05 --mode 'train'
```

## 📊 Testing

```bash
python main.py --dataset-dir '/path/to/dataset' \
               --batch-size 4 \
               --mode 'test' \
               --weight-path '/path/to/weight.tar'
```

**Example:**
```bash
python main.py --dataset-dir './dataset/IRSTD-1K' --batch-size 4 --mode 'test' --weight-path './weight/irstd1k_weight.pkl'
```

## 📈 Results

### Quantitative Results

| Dataset | IoU (×10⁻²) | Pd (×10⁻²) | Fa (×10⁻⁶) | Weights |
|:-------:|:------------:|:----------:|:----------:|:-------:|
| IRSTD-1K | 70.11 | 95.92 | 8.43 | [Download]([https://drive.google.com/file/d/1KqlOVWIktfrBrntzr53z1eGnrzjWCWSe/view?usp=sharing](https://drive.google.com/file/d/1IK__ulzS4kVt6Jtzk3Ljx3AL-jVxKnfS/view?usp=drive_link) |
| NUAA-SIRST | 79.42 | 100.00 | 3.90 | [Download]([https://drive.google.com/file/d/13JQ3V5xhXUcvy6h3opKs15gseuaoKrSQ/view?usp=sharing](https://drive.google.com/file/d/1IK__ulzS4kVt6Jtzk3Ljx3AL-jVxKnfS/view?usp=drive_link) |
| SIRST-Aug | 75.60 | 99.45  | 15.15 | [Download]([https://drive.google.com/file/d/1lcmTgft0LStM7ABWDIMRHTkcOv95p9LO/view?usp=sharing](https://drive.google.com/file/d/1IK__ulzS4kVt6Jtzk3Ljx3AL-jVxKnfS/view?usp=drive_link) |


## 📂 Project Structure

```
MDAFNet/
├── dataset/          # Dataset loading and preprocessing
├── model/            # Network architecture
├── utils/            # Utility functions
├── weight/           # Pretrained weights
├── main.py           # Main entry point
├── requirements.txt  # Dependencies
└── README.md
```

## 🙏 Acknowledgement

We sincerely thank the following works for their contributions:

- [BasicIRSTD](https://github.com/XinyiYing/BasicIRSTD) - A comprehensive toolbox 
- [MSHNet](https://github.com/ying-fu/MSHNet) - Scale and Location Sensitive Loss

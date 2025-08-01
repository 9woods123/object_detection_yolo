# YOLOv8 环境配置指南（含RTX 5080显卡支持）

## 1. 环境准备

### 创建虚拟环境并安装依赖
```bash
# 安装虚拟环境工具（如未安装）
pip install virtualenv
sudo apt install python3.9-venv

# 创建Python 3.9+虚拟环境（RTX 5080需要Python 3.9+）
python -m venv yolo_venv
# 对于Ubuntu系统可能需要额外安装

# 激活虚拟环境
source yolo_venv/bin/activate  # Linux/Mac
# yolo_venv\Scripts\activate   # Windows

# 升级pip到最新版本
pip install --upgrade pip
```

### 安装PyTorch（特别针对RTX 5080）
由于RTX 5080采用Ada Lovelace架构（计算能力sm_120），需要特殊版本的PyTorch：

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

## 2. 安装YOLO依赖

```bash
# 从requirements.txt安装其他依赖
pip install -r requirements.txt
```



### Dataset Preparation and Training

1. **Data Format Conversion**  
   The dataset is stored in the `dataset/` directory and contains the raw data annotated using **LabelMe** (not yet split into training and testing sets).  
   To convert the **LabelMe** annotated dataset into a format that YOLO can read, run the `voc_label.py` script:

   ```bash
   cd dataset 
   python3 voc_label.py
   ```

2. **Split the Dataset**  
   Use the `split_data.py` script to split the dataset into training and testing sets, which will generate a `train_data` folder.

   ```bash
   python3 split_data.py
   ```

3. **Train the Model**  
   After splitting the dataset, use the `train_model.py` script to train the model and generate a `.pt` weight file.

   ```bash
   python3 train_model.py
   ```

4. **Custom Labels**  
   If you want to customize the labels, refer to the `train_cfg.yaml` and `voc_label.py` files for configuration.

---



## Set-up env

```bash
cd Desktop
python3 -m venv demo
source demo/bin/activate
```

## YOLO

### Install
```bash
pip install ultralytics
```

### Split dataset

#### Install scikit-learn
```bash
pip install scikit-learn
```

#### Code to split
```bash
python3 path/to/file/split.py
```

### YOLO usage (CLI)

`yolo [mode] [task]`
with :
- mode = detect (bbox), segment (polygons), classify, pose (keypoints)
- task = train, predict

```bash
yolo detect predict source=/path/to/image/or/path/to/folder_images model=path/to/model
```

```bash
yolo detect train data=config.yaml
```

## LabelStudio

### Install
```bash
pip install label-studio
```

### Start
```bash
label-studio
```
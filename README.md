## KD-FSR:  Towards Efficient Multi-label Image Recognition
#### train and test a baseline on PASCAL VOC 
```
python train_base.py --config_file ../configs/voc/voc07_r34_112_baseline.yaml GPU_ID [0]

change settings "BASE" and "DATA" in voc07_test_baseline.yaml

python test_base.py --config_file ../configs/voc07_test_baseline.yaml GPU_ID [0]
```

#### train and test knowledge distillation on PASCAL VOC 
```
python train_base.py --config_file ../configs/voc/voc07_r101-r34_224-112_kd.yaml GPU_ID [0]

change settings "BASE" and "DATA" of voc07_test_baseline.yaml

python train_kd-2size.py --config_file ../configs/voc07_test_baseline.yaml GPU_ID [0]
```


#### train and test KD-FSR on PASCAL VOC 
```
python train_base.py --config_file ../configs/voc/voc07_r101-r34_224-112_kd-fsr.yaml GPU_ID [0]

change settings "FSR" and "DATA" of voc07_test_fsr.yaml

python train_kd-fsr-2size.py --config_file ../configs/voc07_test_fsr.yaml GPU_ID [0]
```

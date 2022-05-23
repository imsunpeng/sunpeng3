

### 1. preprocess

```bash
RUN preprocess_KiTS.py
```


### 2. Conversion data

```bash
RUN conversion_data.py
```

### 3. Train
```bash
RUN train.py 
```


### 4. Evaluation Test Case
```bash
python eval_agdense_unet.py 
```

### 5. Post-processing
```bash
run post_processing.py 
```

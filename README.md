# Handwritten Text Recognition with Feature Enhancement

English handwritten text recognition on the IAM database using **CRNN-FE**:  
CNN → BiLSTM → **HRNN + Attention** (feature enhancement) → CTC → NLP correction.

![Inigo Montoya](./samples/inigo_montoya1.png)
![Inigo Montoya](./samples/inigo_montoya2.png)

## Pipeline

```
Input → OpenCV Preprocessing → CNN → BiLSTM → HRNN+Attention → CTC → NLP → Output
```

## Requirements

```bash
pip install -r requirements.txt
```

- TensorFlow 1.x, OpenCV, symspellpy, numpy, easydict, tqdm, matplotlib, Pillow  
- CRNN-FE uses `tf.nn.ctc_loss` (no WarpCTC)

## Predict

1. Put images in `samples/`
2. Place `best_model` under `experiments/CRNN_FE_h128/`
3. From `mains/`:

```bash
python predict.py -c ../configs/config.json [--no-nlp] [--clean]
```

- `--no-nlp`: skip spell correction  
- `--clean`: remove `samples/processed/` after run  

## Train

1. **IAM data:** place `lines/` and `lines.txt` in `data/IAM/` (and `aachen_partition/`).
2. From `data/`:

```bash
python preprocess_images.py -c ../configs/config.json
python prepare_IAM.py -c ../configs/config.json
```

3. From `mains/`:

```bash
python main.py -c ../configs/config.json
```

## Eval

From `mains/`:

```bash
python eval.py -c ../configs/config.json
```

## Layout

See `PROJECT_STRUCTURE.md`.

## Citations

- [Laia](https://github.com/jpuigcerver/Laia) — Joan Puigcerver et al.
- Joan Puigcerver. [Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?](http://www.jpuigcerver.net/pubs/jpuigcerver_icdar2017.pdf)
- U. Marti and H. Bunke. The IAM-database. Int. Journal on Document Analysis and Recognition, 5:39–46, 2002.

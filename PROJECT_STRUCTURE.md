# Project Structure — Feature Enhancement Pipeline

```
Handwritten Input → OpenCV Preprocessing → CNN → BiLSTM → HRNN+Attention → CTC → NLP → Output
```

## Layout

```
Handwritten-Text-Recognition/
├── configs/
│   └── config.json           # CRNN-FE config
├── base/
│   ├── base_model.py
│   └── base_train.py
├── data/
│   ├── IAM/
│   │   ├── lines/            # Raw IAM line images
│   │   ├── lines_h128/       # Preprocessed (from preprocess_images.py)
│   │   └── aachen_partition/
│   ├── prepare_IAM.py        # TFRecords + char_map
│   └── preprocess_images.py  # OpenCV: grayscale, blur, Otsu, deskew, resize
├── data_loader/
│   └── data_generator.py
├── mains/
│   ├── main.py               # Train
│   ├── eval.py               # Eval on test set
│   └── predict.py            # OpenCV + CRNN-FE + NLP
├── models/
│   ├── __init__.py
│   ├── CRNN_FE_model.py      # CNN → BiLSTM → HRNN+Attn → CTC
│   └── hrnn_attention.py     # Feature Enhancement
├── preprocessing/
│   ├── __init__.py
│   └── opencv_preprocess.py
├── nlp/
│   ├── __init__.py
│   └── spell_corrector.py
├── trainers/
│   └── trainer.py
├── utils/
│   ├── __init__.py, augment.py, config.py, dirs.py, logger.py, utils.py
├── samples/                  # Input images for predict
├── requirements.txt
├── PROJECT_STRUCTURE.md
└── README.md
```

## Commands

```bash
pip install -r requirements.txt
```

**Data (from `data/`):**
```bash
python preprocess_images.py -c ../configs/config.json
python prepare_IAM.py -c ../configs/config.json
```

**Train / Eval / Predict (from `mains/`):**
```bash
python main.py -c ../configs/config.json
python eval.py -c ../configs/config.json
python predict.py -c ../configs/config.json [--no-nlp] [--clean]
```

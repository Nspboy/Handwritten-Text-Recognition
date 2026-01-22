"""
Predict: OpenCV Preprocessing → CRNN-FE (CNN→BiLSTM→HRNN+Attention→CTC) → NLP Correction

- Reads images from ../samples/
- Writes preprocessed to ../samples/processed/ (OpenCV: grayscale, blur, Otsu, deskew, resize)
- Runs CRNN_FE model, then spell correction
- Use --clean to remove samples/processed after run, --no-nlp to skip NLP
"""

import sys
from pathlib import Path

sys.path.extend(['..'])

from utils.tf_compat import tf
import glob
import numpy as np
from tqdm import tqdm
from utils.config import process_config
from importlib import import_module
from data_loader.data_generator import DataGenerator
from preprocessing.opencv_preprocess import OpenCVPreprocessor
from nlp.spell_corrector import load_corrector
import cv2


def preprocess_samples_opencv(samples_dir, processed_dir, target_height=128):
    samples_dir = Path(samples_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    preproc = OpenCVPreprocessor(target_height=target_height, blur_type='gaussian', deskew=True, invert=True)
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    files = []
    for e in exts:
        files.extend(samples_dir.glob(e))
    files = sorted(set(files))
    for p in files:
        try:
            out = preproc(str(p))
            out_path = processed_dir / (p.stem + '.jpg')
            cv2.imwrite(str(out_path), out)
        except Exception as ex:
            print('Skip {}: {}'.format(p.name, ex))
    return len(files)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', default='../configs/config.json', help='Config JSON')
    ap.add_argument('--clean', action='store_true', help='Remove samples/processed after run')
    ap.add_argument('--no-nlp', action='store_true', help='Skip NLP spell correction')
    args = ap.parse_args()
    try:
        config = process_config(args.config)
    except Exception:
        print('Use: -c ../configs/config.json')
        sys.exit(1)

    root = Path(__file__).resolve().parents[1]
    samples_dir = root / 'samples'
    processed_dir = root / 'samples' / 'processed'

    print('Preprocessing (OpenCV: grayscale, blur, Otsu, deskew, resize) ...')
    n = preprocess_samples_opencv(samples_dir, processed_dir, config.im_height)
    if n == 0:
        print('No images in samples/. Add .png/.jpg and re-run.')
        sys.exit(1)
    print('Processed {} images -> {}'.format(n, processed_dir))

    model_module = import_module('models.' + config.architecture + '_model')
    Model = getattr(model_module, 'Model')
    sess = tf.Session()
    data_loader = DataGenerator(config, eval_phase=True, eval_on_test_data=False)
    model = Model(data_loader, config)
    model.load(sess, config.best_model_dir)

    x, length, lab_length, y, is_training = tf.get_collection('inputs')
    pred = model.prediction
    data_loader.initialize(sess, is_train=False)

    predictions = []
    for _ in tqdm(range(data_loader.num_iterations_val), desc='Predict'):
        res = sess.run([pred], feed_dict={is_training: False})
        sp = res[0][0][0]
        pred_mat = np.zeros(sp.dense_shape)
        for idx, val in enumerate(sp.indices):
            pred_mat[val[0]][val[1]] = sp.values[idx]
        for i in range(pred_mat.shape[0]):
            s = ''.join([data_loader.char_map_inv.get(int(j), '') for j in pred_mat[i]])
            predictions.append(s)

    if not args.no_nlp:
        corrector = load_corrector(use_symspell=True, use_bert=False)
        predictions = [corrector.correct(p) for p in predictions]

    filenames = sorted(glob.glob(str(processed_dir / '*')))
    filenames = [f for f in filenames if Path(f).suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    predictions = predictions[:len(filenames)]
    filenames = filenames[:len(predictions)]

    print('\nPredictions:')
    for path, p in zip(filenames, predictions):
        print('  {}: {}'.format(Path(path).name, p))

    if args.clean and processed_dir.exists():
        import shutil
        shutil.rmtree(processed_dir)
        print('Removed', processed_dir)


if __name__ == '__main__':
    main()

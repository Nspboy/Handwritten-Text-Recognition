"""
OpenCV image preprocessing for IAM: grayscale → blur → Otsu → deskew → resize.
Writes to lines_h{height}/ for prepare_IAM.py.

Run from data/:  python preprocess_images.py -c ../configs/config.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm
from utils.utils import get_args
from utils.config import process_config
from preprocessing.opencv_preprocess import OpenCVPreprocessor
import cv2


def process_ims(input_dir, output_dir, out_height, blur_type='gaussian', deskew=True):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preproc = OpenCVPreprocessor(target_height=out_height, blur_type=blur_type, deskew=deskew, invert=True)
    files = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))
    for p in tqdm(files, desc='Preprocess'):
        try:
            out = preproc(str(p))
            out_path = output_dir / (p.stem + '.jpg')
            cv2.imwrite(str(out_path), out)
        except Exception as e:
            print('Skip {}: {}'.format(p, e))


if __name__ == '__main__':
    try:
        args = get_args()
        config = process_config(args.config)
    except Exception:
        print('Use: -c ../configs/config.json')
        sys.exit(1)
    dataset = config.dataset
    height = config.im_height
    data_root = Path(__file__).resolve().parent
    in_dir = data_root / dataset / 'lines'
    out_dir = data_root / dataset / ('lines_h' + str(height))
    if not in_dir.exists():
        print('Input dir not found:', in_dir)
        sys.exit(1)
    process_ims(in_dir, out_dir, height)
    print('Done. Output:', out_dir)

import os
import glob
import shutil
from PIL import Image
import argparse
import time as t
from utils import mk_dir, unzip
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Preprocess Diagnostic-Kit Image')
parser.add_argument('--root', type=str, default='/home/server1/workdir/all_dataset/Prof_LJH/time-series-data/for_test')
parser.add_argument('--flag', type=dict, default={'unzip': False,
                                                  'relocate': False,
                                                  'downscale': False,
                                                  'preprocess': True })
parser.add_argument('--dir_zip', type=str, default='temp', help='path to zip files')    # TODO
parser.add_argument('--dir_raw', type=str, default='raw', help='path to save decompressed images')
parser.add_argument('--dir_downscale', type=str, default='downscale', help='path to save downscaled images')
parser.add_argument('--dir_crop', type=str, default='crop', help='path to save preprocessed(crop) images')

parser.add_argument('--rotate_angle', type=int, default=270)
parser.add_argument('--crop_size', type=tuple, default=(0.35, 0.116, 0.35, 0.5),
                    help='(left, top, right, bottom)  중앙지점 기준으로 몇 퍼센트 사용할건지 (각 요소당 최대값 : 0.5)')
parser.add_argument('--resize_width', type=int, default=200, help='output image shape : (200,?)')

parser.add_argument('--show_sample_img', type=bool, default=True, help='flag to show processed sample image')
args = parser.parse_args()


def unzip_all(input_dir, output_dir):
    zip_files = glob.glob('{}/*.zip'.format(input_dir))
    for zip_file in tqdm(zip_files, desc='unzip'):
        unzip(zip_file_path=zip_file, save_dir=output_dir)


def relocation(path_):
    density_dir_dict = {}
    for directory in tqdm(os.listdir(path_), desc='relocate'):
        try:
            dir_name = os.path.split(directory)[-1]
            density, idx = dir_name.split('_')
            density = str(int(float(density)*10000))
            if density not in density_dir_dict:
                density_dir_dict[density] = os.path.join(path_, density)
                mk_dir(density_dir_dict[density])
            shutil.move(os.path.join(path_, directory),
                        os.path.join(density_dir_dict[density], '{}_{}'.format(density, idx)))
        except Exception as e:
            print(e, ': invalid directory -', directory)


def downscaling(root_input_dir, root_output_dir):
    density_dict = {}
    mk_dir(root_output_dir)
    for i, (path, dirs, files) in enumerate(os.walk(root_input_dir)):
        if i == 0 or dirs == [] or files != []: continue
        density = os.path.split(path)[1]
        density_dict[density] = len(dirs)
        mk_dir(os.path.join(root_output_dir, density))

        for d in tqdm(sorted(dirs), desc='downscale - density:{}'.format(density)):
            input_dir = os.path.join(path, d)
            output_dir = os.path.join(root_output_dir, density, '{}_{}'.format(density, d.split('_')[-1]))
            downscale_per_dir(input_dir, output_dir, args)
    return density_dict


def preprocessing(root_input_dir, root_output_dir):
    density_dict = {}
    root_output_dir = mk_dir(os.path.join(root_output_dir, 'all'))
    for i, (path, dirs, files) in enumerate(os.walk(root_input_dir)):
        if i == 0 or dirs == [] or files != []: continue
        density = os.path.split(path)[1]
        density_dict[density] = len(dirs)
        mk_dir(os.path.join(root_output_dir, density))

        for d in tqdm(sorted(dirs), desc='preprocess - density:{}'.format(density)):
            input_dir = os.path.join(path, d)
            output_dir = os.path.join(root_output_dir, density, '{}_{}'.format(density, d.split('_')[-1]))
            preprocess_per_dir(input_dir, output_dir, args)
    return density_dict


def downscale_per_dir(input_dir, output_dir, args):
    if not os.path.isdir(output_dir) or len(os.walk(output_dir).__next__()[2]) < 90:
        mk_dir(output_dir)
        files = sorted(glob.glob('{}/*.png'.format(input_dir)))
        for i, f in enumerate(files, 1):
            f_dir, f_name = os.path.split(f)
            preprocessed_img = process_per_img(img_path=f,
                                               rotate=False, rotate_angle=0,
                                               crop=False, crop_size=None,
                                               resize=True, resize_width=300)
            preprocessed_img.save(os.path.join(output_dir, f_name))


def preprocess_per_dir(input_dir, output_dir, args):
    if not os.path.isdir(output_dir) or len(os.walk(output_dir).__next__()[2]) < 90:
        mk_dir(output_dir)
        files = sorted(glob.glob('{}/*.png'.format(input_dir)))
        for i, f in enumerate(files, 1):
            f_dir, f_name = os.path.split(f)
            preprocessed_img = process_per_img(img_path=f,
                                               rotate=False, rotate_angle=args.rotate_angle,
                                               crop=True, crop_size=args.crop_size,
                                               resize=True, resize_width=args.resize_width)
            preprocessed_img.save(os.path.join(output_dir, '{}{}'.format(i*10, os.path.splitext(f_name)[-1])))


def process_per_img(img_path, rotate, rotate_angle, crop, crop_size, resize, resize_width):
    img = Image.open(img_path)
    img = rotate_img(input_img=img,
                     rotate_angle=rotate_angle) if rotate else img
    img = crop_img(input_img=img,
                   crop_size=crop_size) if crop else img
    img = resize_img(input_img=img,
                     resize_width=resize_width) if resize else img
    if args.show_sample_img: img.show()
    args.show_sample_img = False
    return img


def rotate_img(input_img, rotate_angle=0):
    if rotate_angle not in [0, 360]:
        angle = {90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}
        out_img = input_img.transpose(angle[rotate_angle])
        return out_img


def resize_img(input_img, resize_width=200):
    size = (resize_width, resize_width * input_img.height // input_img.width)
    out_img = input_img.resize(size)
    return out_img


def crop_img(input_img, crop_size):
    w, h = input_img.size
    cx, cy = w // 2, h // 2  # center point
    crop_area = (cx - w * crop_size[0], cy - h * crop_size[1], cx + w * crop_size[2],
                 cy + h * crop_size[3])  # (x_start, y_start, x_end, y_end)
    out_img = input_img.crop(crop_area)
    return out_img


if __name__ == '__main__':
    start = time = t.time()

    args.dir_zip = os.path.join(args.root, args.dir_zip)
    args.dir_raw = os.path.join(args.root, args.dir_raw)
    args.dir_downscale = os.path.join(args.root, args.dir_downscale)
    args.dir_crop = os.path.join(args.root, args.dir_crop)

    flag = args.flag

    if flag['unzip']:
        time = t.time()
        unzip_all(input_dir=args.dir_zip, output_dir=args.dir_raw)
        print('\n- unzip ({:.0f} sec) >> {}'.format((t.time() - time), args.dir_raw))

    if flag['relocate']:
        time = t.time()
        relocation(path_=args.dir_raw)
        print('\n- relocation ({:.0f} sec) >> {}'.format((t.time() - time), args.dir_raw))

    if flag['downscale']:
        time = t.time()
        density_dict = downscaling(root_input_dir=args.dir_raw,
                                   root_output_dir=args.dir_downscale)
        print('\n- downscale image ({:.0f} min) >> {}'.format((t.time() - time)/60, args.dir_downscale))

    if flag['preprocess']:
        time = t.time()
        density_dict = preprocessing(root_input_dir=args.dir_downscale,
                                     root_output_dir=args.dir_crop)
        print('\n- preprocess image ({:.0f} min) >> {}'.format((t.time() - time)/60, args.dir_crop))

    print('\n\n - total time: {:.0f} min'.format((t.time() - start)/60))
    if any((flag['downscale'], flag['preprocess'])):
        print(' - Data:', sorted([(int(k),v) for k, v in density_dict.items()]))


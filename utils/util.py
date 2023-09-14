import os
from zipfile import ZipFile, ZIP_DEFLATED


def mk_dir(path_: str):
    os.makedirs(path_) if not os.path.isdir(path_) else None
    return path_


def unzip(zip_file_path: str, save_dir: str=None):
    """
    :param zip_file_path: Path to zip file to decompress
    :param save_dir: Path to save decompressed files
    """
    if not save_dir: save_dir = os.path.split(zip_file_path)[0]
    with ZipFile(zip_file_path, 'r') as zip_file:
        zip_info = zip_file.NameToInfo.copy()
        for k, v in zip_info.items():
            name = v.filename.encode('cp437').decode('euc-kr')
            if k == name: continue
            zip_file.NameToInfo[k].filename = name
            zip_file.NameToInfo[name] = zip_file.NameToInfo[k]
            del zip_file.NameToInfo[k]
        zip_file.extractall(save_dir)


def zip(input_dir: str, save_path: str=None, zip_file_name: str=None):
    """
    :param input_dir: Directory path to convert to zip file
    :param save_path: Path to save zip file
    :param zip_file_name: Name of zip file (Ex. temp.zip)
    """
    if not save_path: save_path = os.path.split(input_dir)[0]
    if not zip_file_name: zip_file_name = os.path.split(input_dir)[1]+'.zip'

    with ZipFile(os.path.join(save_path, zip_file_name), 'w') as zip_file:
        for (path, dirs, files) in os.walk(input_dir):
            for file in files:
                zip_file.write(filename=os.path.join(path, file),
                               arcname=os.path.relpath(os.path.join(path,file), input_dir),
                               compress_type=ZIP_DEFLATED)

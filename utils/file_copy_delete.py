import os
import shutil
from multiprocessing import Pool


def copy_file_to_folder(_target_file):
    source_file = r'Q:\pychem_project\XUT-GeoIFCNet-Master\results\IfcSchemaGraph_embedding\55_FORWARD_IFCSchemaGraph_HGAT_128_512_128_nce.bin'

    # 如果目标文件夹中不存在相同的文件，则复制
    if not os.path.exists(_target_file):
        shutil.copy(source_file, _target_file)
        print("Copied to {}".format(_target_file))
    else:
        print("File already exists in {}".format(_target_file))


def delete_file_from_folder(_target_file):

    if os.path.exists(_target_file):
        os.remove(_target_file)
        print("Deleted {}".format(_target_file))
    else:
        print("File does not exist in {}".format(_target_file))


if __name__ == '__main__':
    # 目标文件夹列表
    target_file = []
    for root, dirs, files in os.walk(r'Q:\IFCNET_TEST\IFCNET_GRAPH'):
        for file in files:
            if file.endswith('node.csv'):
                target_file.append(os.path.join(root, '55_REVERSE_IFCSchemaGraph_HGAT_128_512_128_nce.bin'))

    # 创建进程池
    num_processes = 30
    with Pool(num_processes) as pool:
        pool.map(delete_file_from_folder, target_file)

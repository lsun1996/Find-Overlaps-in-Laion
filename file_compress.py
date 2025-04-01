import os
import shutil

def compress_folder(dir_to_compress, output_zip):
    if not os.path.isdir(dir_to_compress):
        raise ValueError(f"The provided folder path {dir_to_compress} does not exist or is not a directory!")
    shutil.make_archive(output_zip, 'zip', dir_to_compress)
    print(f"Folder {dir_to_compress} compressed successfully into {output_zip}")

if __name__ == "__main__":
    # folder = "/teamspace/studios/this_studio/data/intermediate/food101-train"
    # compress_folder(folder, "food101-train-clip-plots")
    folder = "/teamspace/studios/this_studio/data/intermediate/cub-test"
    compress_folder(folder, "cub-test")

import os


def get_matlab_test_files(path) -> [str]:
    files = []
    for file_name in os.listdir(path):
        if file_name.endswith(".mat"):
            files.append(os.path.join(path, file_name))
    return sorted(files)

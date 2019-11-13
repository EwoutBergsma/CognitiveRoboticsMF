import subprocess


def get_vhf_representation(file_path):
    """
    This function calls a C++ executable to calculate a VHF representation for a given pointcloud filename and parses
    its output to a python list.
    :param file_path: File path to a .pcd file
    :return: The 308 numbers long list representing the VHF histogram of the pointcloud
    """
    args = ("./CPP/build/project", file_path)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read().decode("utf-8")
    try:
        return eval(output)
    except SyntaxError:
        return None

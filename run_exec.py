import subprocess


def get_vhf_representation(file_path):
    args = ("./CPP/build/project", file_path)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read().decode("utf-8")
    try:
        return eval(output)
    except SyntaxError:
        return None

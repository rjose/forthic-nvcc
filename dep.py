import subprocess

def cmd(string):
    result = subprocess.check_output([string], shell=True).decode('utf-8')[:-1]
    return result


def dot_h_files():
    return cmd("ls *.h").split("\n")


def dot_cpp_files():
    return cmd("ls *.cpp").split("\n")


def header_dependencies(header):
    lines = cmd("grep {0} *.cpp".format(header)).split("\n")

    def dot_o(line):
        base = line.split(".cpp")[0]
        return base + ".o"

    dot_os = [dot_o(l) for l in lines]

    result = ""
    if dot_os:
        result = "{0} : {1}".format(" ".join(dot_os), header)
    print(result)
    return result


def dependencies():
    return "\n".join([ header_dependencies(h) for h in dot_h_files()])

dependencies()


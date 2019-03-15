import subprocess

def cmd(string):
    try:
        result = subprocess.check_output([string], shell=True).decode('utf-8')[:-1]
    except:
        result = ""
    return result


def dot_h_files():
    return cmd("ls *.h").split("\n")


def dot_cpp_files():
    return cmd("ls *.cpp").split("\n")


def header_dependencies(header):

    def grep_header(extension):
        return cmd("grep {0} *{1}".format(header, extension)).split("\n")

    def dot_o(line, extension):
        base = line.split(extension)[0]
        if not base:
            return ""
        return base + ".o"

    def deps(extension):
        lines = grep_header(extension)
        return [dot_o(l, extension) for l in lines]

    dot_os = deps(".cpp") + deps(".h")

    result = ""
    if dot_os:
        result = "{0} : {1}".format(" ".join(dot_os), header)
    return result


def dependencies():
    return "\n".join([ header_dependencies(h) for h in dot_h_files()])

print(dependencies())


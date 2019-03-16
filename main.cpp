#include <memory>
#include <cstdio>
#include <fstream>

#include "Interpreter.h"
#include "Module.h"
#include "CudaModule.h"

using namespace std;

string load_file(string filename) {
    ifstream infile { filename };
    if (!infile) {
        throw string("Can't open file: ") + filename;
    }

    string result { istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    return result;
}

int main(int c, char* argv[]) {
    try {
        string filename = "CHECK-DIMENSION.forthic";
        if (c >= 2)   filename = argv[1];

        Interpreter interp;
        interp.RegisterModule(shared_ptr<Module>(new CudaModule()));
        interp.Run(load_file(filename));
    }
    catch (const char *message) {
        printf("EXCEPTION: %s\n", message);
    }
    catch (string message) {
        printf("EXCEPTION: %s\n", message.c_str());
    }
    return 0;
}

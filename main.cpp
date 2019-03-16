#include <memory>
#include <cstdio>
#include "Interpreter.h"
#include "Module.h"
#include "CudaModule.h"

using namespace std;

int main() {
    try {
        Interpreter interp;
        interp.RegisterModule(shared_ptr<Module>(new CudaModule()));
        interp.Run("[ cuda ] USE-MODULES");
        interp.Run("1 15 HELLO");
    }
    catch (const char *message) {
        printf("EXCEPTION: %s\n", message);
    }
    catch (string message) {
        printf("EXCEPTION: %s\n", message.c_str());
    }
    return 0;
}

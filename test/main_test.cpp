#include <cstdio>
#include "TokenizerTest.h"
#include "ModuleTest.h"
#include "InterpreterTest.h"

int main() {
    TokenizerTest tokenizerTest;
    ModuleTest moduleTest;
    InterpreterTest interpTest;

    tokenizerTest.run();
    moduleTest.run();
    interpTest.run();
    return 0;
}

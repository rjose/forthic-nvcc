#include <cstdio>
#include "Test.h"

Test::Test() {
}

void Test::printFailure(bool failed, const char* file, int line) {
    if (failed)   printf("=> FAIL  %s:%d\n", file, line);
}
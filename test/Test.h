#pragma once

class Test {
public:
    Test();
    void printFailure(bool failure, const char* file, int line);
};
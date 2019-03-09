#pragma once
#include <string>

class ModuleTest {
public:
    ModuleTest();
    void run();

protected:
    void printFailure(bool pass, const char* file, int line);

private:
    void testEmptyModule();
    void testAddWord();
    void testEnsureVariable();
    void testSearchUsingModule();
};

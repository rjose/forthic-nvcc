#include <string>
#include "GlobalModuleTest.h"
#include "../Interpreter.h"
#include "../BasicConverters.h"
#include "../GlobalModule.h"


using namespace std;

GlobalModuleTest::GlobalModuleTest() {
}

void GlobalModuleTest::run() {
    testIntLiteral();
    testFloatLiteral();
    testUsingModules();
}


void GlobalModuleTest::testIntLiteral() {
    Interpreter interp;
    interp.Run("27");
    printFailure(27 != AsInt(interp.StackPop()), __FILE__, __LINE__);
}

void GlobalModuleTest::testFloatLiteral() {
    Interpreter interp;
    interp.Run("27.5");
    printFailure(27.5 != AsFloat(interp.StackPop()), __FILE__, __LINE__);
}

void GlobalModuleTest::testUsingModules() {
    Interpreter interp;
    interp.Run("{sample : HI   'Hello' ; } ");

    // Verify that HI is not in the current module's scope
    bool run_failed = false;
    try {
        interp.Run("HI");
    }
    // TODO: Catch an UnknownWordException
    catch(...) {
        run_failed = true;
    }
    printFailure(run_failed == false, __FILE__, __LINE__);

    // If we use USE-MODULES, we can find HI
    interp.Run("[ sample ] USE-MODULES  HI");
}

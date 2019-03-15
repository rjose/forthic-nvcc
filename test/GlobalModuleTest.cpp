#include <string>
#include "GlobalModuleTest.h"
#include "../Interpreter.h"
#include "../BasicItemGetters.h"
#include "../GlobalModule.h"


using namespace std;

GlobalModuleTest::GlobalModuleTest() {
}

void GlobalModuleTest::run() {
    testIntLiteral();
}


void GlobalModuleTest::testIntLiteral() {
    Interpreter interp;
    interp.Run("27");
    shared_ptr<StackItem> item = interp.StackPop();
    printFailure(27 != ForthicGetInt(item.get()), __FILE__, __LINE__);
}


/*
namespace ForthicLibTests
{
    TEST_CLASS(GlobalModuleTest)
    {
    public:
        Interpreter* interp;

        TEST_METHOD_INITIALIZE(Setup)
        {
            interp = new Interpreter();
        }

        TEST_METHOD_CLEANUP(Teardown)
        {
            delete interp;
        }

        TEST_METHOD(TestIntLiteral)
        {
            interp->Run("27");
            shared_ptr<StackItem> item = interp->StackPop();
            Assert::AreEqual(27, ForthicGetInt(item.get()));
        }

        TEST_METHOD(TestFloatLiteral)
        {
            interp->Run("27.5");
            shared_ptr<StackItem> item = interp->StackPop();
            Assert::IsTrue(fabs(ForthicGetFloat(item.get()) - 27.5) < 0.01);
        }

        TEST_METHOD(TestPop)
        {
            interp->Run("1 2 3 POP");
            shared_ptr<StackItem> item = interp->StackPop();
            Assert::AreEqual(2, ForthicGetInt(item.get()));
        }

        TEST_METHOD(TestUsingModules)
        {
            interp->Run("{sample : HI   'Hello' ; } ");

            // Verify that HI is not in the current module's scope
            Interpreter* i = interp;
            Assert::ExpectException<string>([i]() {i->Run("HI"); });  // [i]() {...} is a lambda expression

            // If we use USE-MODULES, we can find HI
            interp->Run("[ sample ] USE-MODULES  HI");
        }
    };
}
*/

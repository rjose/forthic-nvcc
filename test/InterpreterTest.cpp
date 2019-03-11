#include "InterpreterTest.h"
#include "../Interpreter.h"
#include "../StackItem.h"
// #include "../StringItem.h"
// #include "../ArrayItem.h"
#include <string>

using namespace std;

InterpreterTest::InterpreterTest() {
}

void InterpreterTest::run() {
    try {
        testPushString();
        testPushEmptyArray();
        testPushArray();
    }
    catch (const char *message) {
        printf("EXCEPTION: %s\n", message);
    }
    catch (string message) {
        printf("EXCEPTION: %s\n", message.c_str());
    }
}

void InterpreterTest::testPushString() {
    Interpreter interp;
    interp.Run("'Howdy'");
    shared_ptr<StackItem> item = interp.StackPop();
    printFailure(string("Howdy") != ForthicGetString(item.get()), __FILE__, __LINE__);
}

void InterpreterTest::testPushEmptyArray() {
    Interpreter interp;
    interp.Run("[ ]");
    shared_ptr<StackItem> array_item = interp.StackPop();
    vector<shared_ptr<StackItem>> items = ForthicGetArray(array_item.get());
    printFailure(items.size() != 0, __FILE__, __LINE__);
}

void InterpreterTest::testPushArray() {
    Interpreter interp;
    interp.Run("[ 'One' 'Two' ]");
    shared_ptr<StackItem> array_item = interp.StackPop();
    vector<shared_ptr<StackItem>> items = ForthicGetArray(array_item.get());
    printFailure(2 != (int)items.size(), __FILE__, __LINE__);
    printFailure(string("One") != ForthicGetString(items[0].get()), __FILE__, __LINE__);
    printFailure(string("Two") != ForthicGetString(items[1].get()), __FILE__, __LINE__);
}


/*
namespace ForthicLibTests
{
    TEST_CLASS(InterpreterTest)
    {
    public:

        TEST_METHOD(TestPushModule)
        {
            Interpreter interp;
            interp.Run("{sample");
            auto mod = interp.CurModule();
            Assert::AreEqual(string("sample"), mod->GetName());

            interp.Run("}");
            mod = interp.CurModule();
            Assert::AreEqual(string(""), mod->GetName());
        }

        TEST_METHOD(TestCreateDefinition)
        {
            Interpreter interp;
            interp.Run(": TACO 'taco' ;");
            auto mod = interp.CurModule();
            auto word = mod->FindWord("TACO");
            Assert::AreEqual(string("TACO"), word->GetName());

            // Execute definition
            interp.Run("TACO");
            shared_ptr<StackItem> val = interp.StackPop();
            Assert::AreEqual(string("taco"), ForthicGetString(val.get()));
        }

    };
}
*/
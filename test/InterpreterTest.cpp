#include <string>
#include "InterpreterTest.h"
#include "../Interpreter.h"
#include "../StackItem.h"

using namespace std;

InterpreterTest::InterpreterTest() {
}

void InterpreterTest::run() {
    testPushString();
    testPushEmptyArray();
    testPushArray();
    testPushModule();
    testCreateDefinition();
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


void InterpreterTest::testPushModule() {
    Interpreter interp;
    interp.Run("{sample");
    auto mod = interp.CurModule();
    printFailure(string("sample") != mod->GetName(), __FILE__, __LINE__);

    interp.Run("}");
    mod = interp.CurModule();
    printFailure(string("") != mod->GetName(), __FILE__, __LINE__);
}


void InterpreterTest::testCreateDefinition() {
    Interpreter interp;
    interp.Run(": TACO 'taco' ;");
    auto mod = interp.CurModule();
    auto word = mod->FindWord("TACO");
    printFailure(string("TACO") != word->GetName(), __FILE__, __LINE__);

    // Execute definition
    interp.Run("TACO");
    shared_ptr<StackItem> val = interp.StackPop();
    printFailure(string("taco") != ForthicGetString(val.get()), __FILE__, __LINE__);
}

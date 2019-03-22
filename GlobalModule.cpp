#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include "Interpreter.h"
#include "GlobalModule.h"
#include "PushItemWord.h"
#include "FloatItem.h"
#include "IntItem.h"
#include "StringItem.h"
#include "AddressItem.h"
#include "TimePointItem.h"

// =============================================================================
// Words

// ( a -- )
// Pops word from stack
class PopWord : public Word
{
public:
    PopWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        interp->StackPop();
    }
};


// ( a b -- b a )
// Swaps top two items
class SwapWord : public Word
{
public:
    SwapWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        auto b = interp->StackPop();
        auto a = interp->StackPop();
        interp->StackPush(b);
        interp->StackPush(a);
    }
};


// ( modules -- )
// Adds modules to current module's using module list
class UseModulesWord : public Word
{
public:
    UseModulesWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        auto item = interp->StackPop();
        vector<shared_ptr<StackItem>> modules = AsArray(item);
        for (int i = 0; i < modules.size(); i++) {
            shared_ptr<Module> m = AsModule(modules[i]);
            interp->CurModule()->UseModule(m);
        }
    }
};


// ( names -- )
// Creates variables in current module
class VariablesWord : public Word
{
public:
    VariablesWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        auto names = AsArray(interp->StackPop());

        for (int i = 0; i < names.size(); i++) {
            string name = AsString(names[i]);
            interp->CurModule()->EnsureVariable(name);
        }
    }
};


// ( value variable -- )
// Sets variable value
class BangWord : public Word
{
public:
    BangWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        auto variable_item = interp->StackPop();
        VariableItem* variable = dynamic_cast<VariableItem*>(variable_item.get());
        auto value = interp->StackPop();
        variable->SetValue(value);
    }
};


// ( variable -- value )
// Gets variable value
class AtWord : public Word
{
public:
    AtWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        auto variable_item = interp->StackPop();
        VariableItem* variable = dynamic_cast<VariableItem*>(variable_item.get());
        interp->StackPush(variable->GetValue());
    }
};


// ( -- )
// Prints param stack
class DotSWord : public Word
{
public:
    DotSWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        stack<shared_ptr<StackItem>> temp_stack;

        int stack_size = interp->StackSize();

        if (stack_size == 0) {
            printf("[]\n");
            return;
        }

        for (int i=0; i < stack_size; i++) {
            auto item = interp->StackPop();
            printf("[%d] %s\n", i, item->StringRep().c_str());
            temp_stack.push(item);
        } 

        // Push items back
        for (int i=0; i < stack_size; i++) {
            auto item = temp_stack.top();
            temp_stack.pop();
            interp->StackPush(item);
        } 
    }
};


// ( l r -- res )
class ArithWord : public Word
{
public:
    ArithWord(string name, string op) : Word(name), op(op) {};

    virtual void Execute(Interpreter *interp) {
        float r = AsFloat(interp->StackPop());
        float l = AsFloat(interp->StackPop());

        float res = 0.0;
        int int_res = 0;
        if      (op == "+")    res = l + r;
        else if (op == "-")    res = l - r;
        else if (op == "*")    res = l * r;
        else if (op == "/")    res = l / r;
        else if (op == "<<")   int_res = (int)l << (int)r;
        else                  throw string("Unknown operation: ") + op;

        if (op == "<<") {
            interp->StackPush(shared_ptr<IntItem>(new IntItem(int_res)));
        }
        else {
            interp->StackPush(shared_ptr<FloatItem>(new FloatItem(res)));
        }
    }
protected:
    string op;
};


// ( item -- )
class PrintWord : public Word
{
public:
    PrintWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        auto item = interp->StackPop();
        printf("%s\n", item->AsString().c_str());
    }
};


// ( item -- )
class StopWord : public Word
{
public:
    StopWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        throw "STOP";
    }
};


// ( ms -- )
class MSleepWord : public Word
{
public:
    MSleepWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        int msec = AsInt(interp->StackPop());
        usleep(msec*1000);
    }
};

// ( strings -- string )
class ConcatWord : public Word
{
public:
    ConcatWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        auto strings = AsArray(interp->StackPop());
        string result;
        for (int i=0; i < strings.size(); i++) {
            result += strings[i]->AsString();
        }
        interp->StackPush(StringItem::New(result));
    }
};

// ( -- "\n" )
class EndlWord : public Word
{
public:
    EndlWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        interp->StackPush(StringItem::New("\n"));
    }
};


// ( items str -- )
class ForeachWord : public Word
{
public:
    ForeachWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        string str = AsString(interp->StackPop());
        auto items = AsArray(interp->StackPop());
        for (int i=0; i < items.size(); i++) {
            interp->StackPush(items[i]);
            interp->Run(str);
        }
    }
};



// ( num-bytes -- address )
class MallocWord : public Word
{
public:
    MallocWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        int num_bytes = AsInt(interp->StackPop());
        void* ref = malloc(num_bytes);
        interp->StackPush(shared_ptr<AddressItem>(new AddressItem(ref)));
    }
};


// ( address value num-bytes -- )
class MemsetWord : public Word
{
public:
    MemsetWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        int num_bytes = AsInt(interp->StackPop());
        int value = AsInt(interp->StackPop());
        void* address = AsVoidStar(interp->StackPop());
        memset(address, value, num_bytes);
    }
};


// ( address -- )
class FreeWord : public Word
{
public:
    FreeWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        void* address = AsVoidStar(interp->StackPop());
        free(address);
    }
};

// ( -- time_point )
class NowWord : public Word
{
public:
    NowWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        interp->StackPush(TimePointItem::New(high_resolution_clock::now()));
    }
};

// ( l_time_point r_time_point -- ms )
class SinceWord : public Word
{
public:
    SinceWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        auto r_time_point = AsTimePoint(interp->StackPop());
        auto l_time_point = AsTimePoint(interp->StackPop());
        auto duration = r_time_point - l_time_point;
        long long result = duration_cast<milliseconds>(duration).count();
        interp->StackPush(IntItem::New(result));
    }
};

// =============================================================================
// GlobalModule

GlobalModule::GlobalModule() : Module("Forthic.global")
{
    AddWord(shared_ptr<Word>(new PopWord("POP")));
    AddWord(shared_ptr<Word>(new SwapWord("SWAP")));
    AddWord(shared_ptr<Word>(new UseModulesWord("USE-MODULES")));
    AddWord(shared_ptr<Word>(new VariablesWord("VARIABLES")));
    AddWord(shared_ptr<Word>(new BangWord("!")));
    AddWord(shared_ptr<Word>(new AtWord("@")));
    AddWord(shared_ptr<Word>(new DotSWord(".s")));
    AddWord(shared_ptr<Word>(new ArithWord("+", "+")));
    AddWord(shared_ptr<Word>(new ArithWord("-", "-")));
    AddWord(shared_ptr<Word>(new ArithWord("*", "*")));
    AddWord(shared_ptr<Word>(new ArithWord("/", "/")));
    AddWord(shared_ptr<Word>(new ArithWord("<<", "<<")));
    AddWord(shared_ptr<Word>(new PrintWord("PRINT")));
    AddWord(shared_ptr<Word>(new StopWord("STOP")));
    AddWord(shared_ptr<Word>(new MSleepWord("MSLEEP")));
    AddWord(shared_ptr<Word>(new ConcatWord("CONCAT")));
    AddWord(shared_ptr<Word>(new EndlWord("ENDL")));
    AddWord(shared_ptr<Word>(new ForeachWord("FOREACH")));
    AddWord(shared_ptr<Word>(new MallocWord("MALLOC")));
    AddWord(shared_ptr<Word>(new MemsetWord("MEMSET")));
    AddWord(shared_ptr<Word>(new FreeWord("FREE")));
    AddWord(shared_ptr<Word>(new NowWord("NOW")));
    AddWord(shared_ptr<Word>(new SinceWord("SINCE")));
}


shared_ptr<Word> GlobalModule::treat_as_float(string name)
{
    try {
        float value = stof(name);
        return shared_ptr<Word>(new PushItemWord(name, shared_ptr<StackItem>(new FloatItem(value))));
    }
    catch (...) {
        return nullptr;
    }
}


shared_ptr<Word> GlobalModule::treat_as_int(string name)
{
    try {
        string::size_type sz;
        int value = stoi(name, &sz);
        char c = name[sz];
        if (c == '.' || c == 'e' || c == 'E') return nullptr;
        else  return shared_ptr<Word>(new PushItemWord(name, shared_ptr<StackItem>(new IntItem(value))));
    }
    catch (...) {
        return nullptr;
    }
}


shared_ptr<Word> GlobalModule::treat_as_literal(string name)
{
    shared_ptr<Word> result = nullptr;
    if (result == nullptr)  result = treat_as_int(name);
    if (result == nullptr)  result = treat_as_float(name);
    return result;
}

// =============================================================================
// StackItem Converters

int AsInt(shared_ptr<StackItem> item) {
    if (auto i = dynamic_cast<IGetInt*>(item.get())) {
        return i->GetInt();
    }
    else {
        throw item->StringRep() + " does not implement IGetInt";
    }
}

float AsFloat(shared_ptr<StackItem> item) {
    if (auto i = dynamic_cast<IGetFloat*>(item.get())) {
        return i->GetFloat();
    }
    else {
        throw item->StringRep() + " does not implement IGetFloat";
    }
}

high_resolution_clock::time_point AsTimePoint(shared_ptr<StackItem> item) {
    if (auto i = dynamic_cast<IGetTimePoint*>(item.get())) {
        return i->GetTimePoint();
    }
    else {
        throw item->StringRep() + " does not implement IGetTimePoint";
    }
}

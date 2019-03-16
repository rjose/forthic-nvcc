#include "Interpreter.h"
#include "GlobalModule.h"
#include "PushItemWord.h"
#include "FloatItem.h"
#include "IntItem.h"

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



GlobalModule::GlobalModule() : Module("Forthic.global")
{
    AddWord(shared_ptr<Word>(new PopWord("POP")));
    AddWord(shared_ptr<Word>(new UseModulesWord("USE-MODULES")));
}


GlobalModule::~GlobalModule()
{
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

int AsInt(shared_ptr<StackItem> item)
{
    if (auto i = dynamic_cast<IGetInt*>(item.get()))
    {
        return i->GetInt();
    }
    else
    {
        throw "Item does not implement IGetInt";
    }
}

float AsFloat(shared_ptr<StackItem> item)
{
    if (auto i = dynamic_cast<IGetFloat*>(item.get()))
    {
        return i->GetFloat();
    }
    else
    {
        throw "Item does not implement IGetFloat";
    }
}

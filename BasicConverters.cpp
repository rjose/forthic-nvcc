#include "BasicConverters.h"


string AsString(shared_ptr<StackItem> item)
{
    if (auto i = dynamic_cast<IAsString*>(item.get()))
    {
        return i->AsString();
    }
    else
    {
        throw "Item does not implement IAsString";
    }
}

vector<shared_ptr<StackItem>> AsArray(shared_ptr<StackItem> item)
{
    if (auto i = dynamic_cast<IAsArray*>(item.get()))
    {
        return i->AsArray();
    }
    else
    {
        throw "Item does not implement IAsArray";
    }
}

shared_ptr<VariableItem> AsVariable(shared_ptr<StackItem> item)
{
    if (auto i = dynamic_cast<IAsVariable*>(item.get()))
    {
        return i->AsVariable();
    }
    else
    {
        throw "Item does not implement IAsVariable";
    }
}

shared_ptr<Module> AsModule(shared_ptr<StackItem> item)
{
    if (auto i = dynamic_cast<IAsModule*>(item.get()))
    {
        return i->AsModule();
    }
    else
    {
        throw "Item does not implement IAsModule";
    }
}

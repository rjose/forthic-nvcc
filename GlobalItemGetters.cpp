#include "GlobalItemGetters.h"


string ForthicGetString(StackItem *item)
{
    if (auto i = dynamic_cast<IGetString*>(item))
    {
        return i->GetString();
    }
    else
    {
        throw "Item does not implement IGetString";
    }
}

vector<shared_ptr<StackItem>> ForthicGetArray(StackItem *item)
{
    if (auto i = dynamic_cast<IGetArray*>(item))
    {
        return i->GetArray();
    }
    else
    {
        throw "Item does not implement IGetArray";
    }
}

shared_ptr<StackItem> ForthicGetValue(StackItem *item)
{
    if (auto i = dynamic_cast<IGetValue*>(item))
    {
        return i->GetValue();
    }
    else
    {
        throw "Item does not implement IGetVariable";
    }
}

shared_ptr<Module> ForthicGetModule(StackItem *item)
{
    if (auto i = dynamic_cast<IGetModule*>(item))
    {
        return i->GetModule();
    }
    else
    {
        throw "Item does not implement IGetModule";
    }
}


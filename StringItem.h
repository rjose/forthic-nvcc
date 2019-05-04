#pragma once
#include <string>

#include "StackItem.h"
#include "./m_global/BasicConverters.h"

using namespace std;


class StringItem : public StackItem, public IAsString
{
public:
    StringItem(string s) : item_string(s) {};
    static shared_ptr<StringItem> New(string s);
    virtual ~StringItem() {};
    string AsString();

    virtual string StringRep();

protected:
    string item_string;
};

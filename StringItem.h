#pragma once
#include <string>

#include "StackItem.h"
#include "./m_global/I_AsString.h"

using namespace std;


class StringItem : public StackItem, public I_AsString
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

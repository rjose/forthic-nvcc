#pragma once
#include <string>

#include "BasicConverters.h"
#include "StackItem.h"

using namespace std;


class StringItem : public StackItem, public IAsString
{
public:
	StringItem(string s) : item_string(s) {};
	virtual ~StringItem() {};
	string AsString();

protected:
	string item_string;
};

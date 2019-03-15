#pragma once
#include <string>

#include "BasicItemGetters.h"
#include "StackItem.h"

using namespace std;


class StringItem : public StackItem, public IGetString
{
public:
	StringItem(string s);
	virtual ~StringItem();
	string GetString();

protected:
	string item_string;
};

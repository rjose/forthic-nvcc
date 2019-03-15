#pragma once
#include <string>

#include "GlobalModule.h"
#include "StackItem.h"

using namespace std;


class IntItem : public StackItem, public IGetInt
{
public:
	IntItem(int value);
	virtual ~IntItem();
	int GetInt();

protected:
	int value;
};

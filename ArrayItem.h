#pragma once
#include <memory>
#include <string>
#include <vector>

#include "BasicItemGetters.h"
#include "StackItem.h"

using namespace std;


class ArrayItem : public StackItem, public IGetArray
{
public:
	ArrayItem(vector<shared_ptr<StackItem>> items);
	virtual ~ArrayItem();
	vector<shared_ptr<StackItem>> GetArray();

protected:
	vector<shared_ptr<StackItem>> items;
};

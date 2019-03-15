#pragma once
#include <string>

#include "GlobalModule.h"
#include "StackItem.h"

using namespace std;


class FloatItem : public StackItem, public IGetFloat
{
public:
	FloatItem(float value);
	virtual ~FloatItem();
	float GetFloat();

protected:
	float value;
};

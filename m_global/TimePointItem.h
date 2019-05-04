#pragma once
#include <string>
#include <chrono>

#include "GlobalModule.h"
#include "../StackItem.h"

using namespace std;
using namespace std::chrono;


class TimePointItem : public StackItem, public IGetTimePoint
{
public:
    TimePointItem(high_resolution_clock::time_point value) : value(value) {};
    static shared_ptr<TimePointItem> New(high_resolution_clock::time_point value);

    high_resolution_clock::time_point GetTimePoint();

    virtual string StringRep();
    virtual string AsString();

protected:
    high_resolution_clock::time_point value;
};

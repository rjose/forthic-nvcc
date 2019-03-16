#include <sstream>
#include "IntItem.h"


IntItem::IntItem(int _value) : value(_value)
{
}


IntItem::~IntItem()
{
}

int IntItem::GetInt()
{
    return value;
}

string IntItem::StringRep() {
    stringstream builder;
    builder << "IntItem: " << value;
    return builder.str();
}

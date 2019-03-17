#include <sstream>
#include "FloatItem.h"


FloatItem::FloatItem(float _value) : value(_value)
{
}


FloatItem::~FloatItem()
{
}

float FloatItem::GetFloat() {
    return value;
}

int FloatItem::GetInt() {
    return int(value);
}


string FloatItem::StringRep() {
    stringstream builder;
    builder << "FloatItem: " << value;
    return builder.str();
}

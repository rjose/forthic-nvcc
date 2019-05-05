#include <sstream>
#include "FloatItem.h"


FloatItem::FloatItem(float _value) : value(_value)
{
}


FloatItem::~FloatItem()
{
}

float FloatItem::AsFloat() {
    return value;
}

int FloatItem::AsInt() {
    return int(value);
}


string FloatItem::StringRep() {
    stringstream builder;
    builder << "FloatItem: " << value;
    return builder.str();
}

string FloatItem::AsString() {
    stringstream builder;
    builder << value;
    return builder.str();
}

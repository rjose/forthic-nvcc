#include <sstream>
#include "IntItem.h"


int IntItem::GetInt() {
    return value;
}

float IntItem::GetFloat() {
    return float(value);
}

string IntItem::StringRep() {
    stringstream builder;
    builder << "IntItem: " << value;
    return builder.str();
}

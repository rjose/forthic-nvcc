#include <sstream>
#include "IntItem.h"


shared_ptr<IntItem> IntItem::New(int value) {
    return shared_ptr<IntItem>(new IntItem(value));
}

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

string IntItem::AsString() {
    stringstream builder;
    builder << value;
    return builder.str();
}

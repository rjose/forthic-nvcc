#include <sstream>
#include "ArrayItem.h"

vector<shared_ptr<StackItem>> ArrayItem::AsArray() {
    return items;
}

string ArrayItem::AsString() {
    stringstream builder;
    builder << "ArrayItem(" << items.size() << ")";
    return builder.str();
}

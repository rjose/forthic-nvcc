#include "StringItem.h"

shared_ptr<StringItem> StringItem::New(string s) {
    return shared_ptr<StringItem>(new StringItem(s));
}

string StringItem::AsString() {
    return item_string;
}

string StringItem::StringRep() {
    return item_string;
}

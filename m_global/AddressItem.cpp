#include <sstream>
#include "AddressItem.h"


shared_ptr<AddressItem> AddressItem::New(void* address) {
    return shared_ptr<AddressItem>(new AddressItem(address));
}

float* AddressItem::AsFloatStar() {
    return (float*)(address);
}

int* AddressItem::AsIntStar() {
    return (int*)(address);
}

void* AddressItem::AsVoidStar() {
    return (address);
}

string AddressItem::StringRep() {
    stringstream builder;
    builder << "AddressItem: " << (long int)(address);
    return builder.str();
}

string AddressItem::AsString() {
    stringstream builder;
    builder << (long int)(address);
    return builder.str();
}

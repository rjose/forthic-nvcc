#include <sstream>
#include "AddressItem.h"


shared_ptr<AddressItem> AddressItem::New(void* address) {
    return shared_ptr<AddressItem>(new AddressItem(address));
}

float* AddressItem::GetFloatStar() {
    return (float*)(address);
}

int* AddressItem::GetIntStar() {
    return (int*)(address);
}

void* AddressItem::GetVoidStar() {
    return (address);
}

string AddressItem::StringRep() {
    stringstream builder;
    builder << "AddressItem: " << (long int)(address);
    return builder.str();
}

#include "IGetAddress.h"


float* AsFloatStar(shared_ptr<StackItem> item) {
    if (auto i = dynamic_cast<IGetAddress*>(item.get())) {
        return i->GetFloatStar();
    }
    else {
        throw item->StringRep() + ": does not implement IGetAddress";
    }
}

int* AsIntStar(shared_ptr<StackItem> item) {
    if (auto i = dynamic_cast<IGetAddress*>(item.get())) {
        return i->GetIntStar();
    }
    else {
        throw item->StringRep() + ": does not implement IGetAddress";
    }
}

void* AsVoidStar(shared_ptr<StackItem> item) {
    if (auto i = dynamic_cast<IGetAddress*>(item.get())) {
        return i->GetVoidStar();
    }
    else {
        throw item->StringRep() + ": does not implement IGetAddress";
    }
}

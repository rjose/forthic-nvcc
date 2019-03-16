#include "VariableItem.h"

using namespace std;

shared_ptr<VariableItem> VariableItem::AsVariable() {
    return shared_ptr<VariableItem>(this);
}

shared_ptr<StackItem> VariableItem::GetValue() {
    return value;
}

void VariableItem::SetValue(shared_ptr<StackItem> new_value) {
    value = new_value;
}


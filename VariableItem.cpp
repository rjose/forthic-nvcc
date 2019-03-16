#include "VariableItem.h"

using namespace std;

shared_ptr<StackItem> VariableItem::GetValue() {
    return value;
}

void VariableItem::SetValue(shared_ptr<StackItem> new_value) {
    value = new_value;
}

string VariableItem::StringRep() {
    string value_str = "nullptr";
    if (value != nullptr)   value_str = value->StringRep();
    string result = "VariableItem: ";
    result += value_str;
    return result;
}

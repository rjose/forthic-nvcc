#include <sstream>
#include "../m_global/GlobalModule.h"
#include "LPEquationItem.h"

LPEquationItem::LPEquationItem(vector<shared_ptr<StackItem>> coeff_vals, string name) : name(name) {
    num_coeffs = coeff_vals.size();

    coeffs = (float*)malloc(num_coeffs*sizeof(float));
    if (coeffs == nullptr)   throw "LPEquationItem - malloc failed";

    // Copy values over
    float* cur_val = coeffs;
    for (int i=0; i < coeff_vals.size(); i++) {
        *cur_val++ = AsFloat(coeff_vals[i]);
    }
}

LPEquationItem::~LPEquationItem() {
    free((void*)coeffs);
}

shared_ptr<LPEquationItem> LPEquationItem::New(vector<shared_ptr<StackItem>> items, string name) {
    auto result = shared_ptr<LPEquationItem>(new LPEquationItem(items, name));
    return result;
}


string LPEquationItem::StringRep() {
    stringstream builder;
    builder << "LPEquationItem: " << name;
    return builder.str();
}

string LPEquationItem::AsString() {
    return StringRep();
}


LPEquationItem* AsLPEquationItem(shared_ptr<StackItem> item) {
    if (auto i = dynamic_cast<LPEquationItem*>(item.get())) {
        return i;
    }
    else {
        throw item->StringRep() + " is not an LPEquationItem";
    }
}

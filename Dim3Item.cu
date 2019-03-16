#include <sstream>
#include "Dim3Item.h"


dim3 Dim3Item::GetDim3() {
    return value;
}

string Dim3Item::StringRep() {
    stringstream builder;
    builder << "Dim3Item: " << "(" << value.x << ", " << value.y << ", " << value.z << ")";
    return builder.str();
}
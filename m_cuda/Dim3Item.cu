#include <sstream>
#include "Dim3Item.h"


dim3 Dim3Item::AsDim3() {
    return value;
}

string Dim3Item::StringRep() {
    stringstream builder;
    builder << "Dim3Item: " << "(" << value.x << ", " << value.y << ", " << value.z << ")";
    return builder.str();
}

string Dim3Item::AsString() {
    stringstream builder;
    builder << "(" << value.x << ", " << value.y << ", " << value.z << ")";
    return builder.str();
}

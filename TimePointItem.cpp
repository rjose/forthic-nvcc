#include <sstream>
#include "TimePointItem.h"

shared_ptr<TimePointItem> TimePointItem::New(high_resolution_clock::time_point value) {
    return shared_ptr<TimePointItem>(new TimePointItem(value));
}

high_resolution_clock::time_point TimePointItem::GetTimePoint() {
    return value;
}


string TimePointItem::StringRep() {
    stringstream builder;
    builder << "TimePointItem";
    return builder.str();
}

string TimePointItem::AsString() {
    stringstream builder;
    builder << "TimePointItem";
    return builder.str();
}

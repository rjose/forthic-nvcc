#include "CudaDevicePropItem.h"

shared_ptr<CudaDevicePropItem> CudaDevicePropItem::New(cudaDeviceProp value) {
    return shared_ptr<CudaDevicePropItem>(new CudaDevicePropItem(value));
}

const cudaDeviceProp& CudaDevicePropItem::deviceProp() {
    return value;
}

string CudaDevicePropItem::StringRep() {
    return "CudaDevicePropItem";
}

string CudaDevicePropItem::AsString() {
    return "CudaDevicePropItem";
}

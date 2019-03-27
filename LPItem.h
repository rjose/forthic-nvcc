#pragma once
#include <string>

#include "CudaModule.h"
#include "StackItem.h"

using namespace std;


class LPItem : public StackItem
{
public:
    LPItem(Interpreter* interp);

    virtual ~LPItem() {};
    void Free();

    void PrintMatrix();

    virtual string StringRep();
    virtual string AsString();

protected:

    void allocateMatrixMemory();
    void fillMatrixMemory();
    void fillConstraint(int constraintIndex);

protected:
    Interpreter* interp;

    vector<shared_ptr<StackItem>> constraints;
    shared_ptr<StackItem> objective;
    vector<shared_ptr<StackItem>> varnames;

    int num_cols;
    int num_rows;
    int num_elems;
    float *matrix;
};


LPItem* AsLPItem(shared_ptr<StackItem> item);

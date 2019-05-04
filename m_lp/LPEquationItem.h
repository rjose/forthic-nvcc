#pragma once
#include <string>

#include "../m_cuda/CudaModule.h"
#include "../StackItem.h"
#include "../ArrayItem.h"

using namespace std;


class LPEquationItem : public StackItem
{
public:
    LPEquationItem(vector<shared_ptr<StackItem>> coeffs, string name);
    virtual ~LPEquationItem();

    static shared_ptr<LPEquationItem> New(vector<shared_ptr<StackItem>> coeffs, string name);

    string GetName() { return name; }

    virtual string StringRep();
    virtual string AsString();

    int NumCoeffs() { return num_coeffs; }
    const float* Coeffs() { return coeffs; }

protected:
    string name;
    int num_coeffs;
    float* coeffs;
};


LPEquationItem* AsLPEquationItem(shared_ptr<StackItem> item);

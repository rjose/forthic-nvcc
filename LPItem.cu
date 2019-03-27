#include <sstream>
#include "CudaModule.h"
#include "LPItem.h"
#include "LPEquationItem.h"
#include "Interpreter.h"
#include "FloatItem.h"
#include "IntItem.h"
#include "AddressItem.h"
#include "LPEquationItem.h"


// ( varnames objective constraints  )
LPItem::LPItem(Interpreter* interp) : interp(interp) {
    constraints = AsArray(interp->StackPop());
    objective = interp->StackPop();
    varnames = AsArray(interp->StackPop());

    // structural + logical + RHS
    num_cols = varnames.size() + constraints.size() + 1;

    // objective + constraints
    num_rows = 1 + constraints.size();

    num_elems = num_rows * num_cols;

    // Allocate memory
    allocateMatrixMemory();
    // TODO: Allocate ratio memory

    fillMatrixMemory();
}


void LPItem::Free() {
    interp->StackPush(AddressItem::New((void*)matrix));
    interp->Run("CUDA-FREE");

    // TODO: Free ratio memory
}


void LPItem::PrintMatrix() {
    interp->StackPush(shared_ptr<IntItem>(new IntItem(num_rows)));
    interp->StackPush(shared_ptr<IntItem>(new IntItem(num_cols)));
    interp->StackPush(AddressItem::New((void*) matrix));
    interp->Run("PRINT-MATRIX");
}


void LPItem::allocateMatrixMemory() {
    int num_bytes = num_elems * sizeof(float);
    interp->StackPush(shared_ptr<IntItem>(new IntItem(num_bytes)));
    interp->Run("CUDA-MALLOC-MANAGED");
    matrix = AsFloatStar(interp->StackPop());
    for (int i=0; i < num_elems; i++)   matrix[i] = 0.0;
}


void LPItem::fillMatrixMemory() {
    int col = 0;

    // Objective
    auto obj_eq = AsLPEquationItem(objective);
    int num_coeffs = obj_eq->NumCoeffs();
    const float* coeffs = obj_eq->Coeffs();
    for (int i=0; i < num_coeffs; i++)   matrix[col++] = coeffs[i];

    // Constraints
    for (int i=0; i < constraints.size(); i++)   fillConstraint(i);
}


void LPItem::fillConstraint(int constraintIndex) {
    int col=0;
    int offset = (constraintIndex + 1) * num_cols;  // Objective is in the first row
    auto constraint_eq = AsLPEquationItem(constraints[constraintIndex]);
    int num_coeffs = constraint_eq->NumCoeffs();
    const float* coeffs = constraint_eq->Coeffs();

    // Add structural coeffs
    for (int i=0; i < num_coeffs; i++)    matrix[offset+col++] = coeffs[i];

    // Add logical coeff
    matrix[offset+col+constraintIndex] = 1.0;
}


string LPItem::StringRep() {
    stringstream builder;
    builder << "LPItem";
    return builder.str();
}

string LPItem::AsString() {
    return StringRep();
}



LPItem* AsLPItem(shared_ptr<StackItem> item) {
    if (auto i = dynamic_cast<LPItem*>(item.get())) {
        return i;
    }
    else {
        throw item->StringRep() + " is not an LPItem";
    }
}

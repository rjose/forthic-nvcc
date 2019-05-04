#include <ctime>
#include <cstdio>
#include <cmath>
#include "../Interpreter.h"

#include "../m_global/IntItem.h"
#include "../m_global/ArrayItem.h"
#include "../m_global/AddressItem.h"
#include "../m_global/I_AsString.h"

#include "../m_cuda/CudaModule.h"
#include "../m_cuda/Dim3Item.h"

#include "LinearProgramModule.h"
#include "LPEquationItem.h"
#include "LPItem.h"


// =============================================================================
// Kernels



// =============================================================================
// Words


// ( coeffs name  -- LPEquationItem )
class W_LPEqn : public Word
{
public:
    W_LPEqn(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        string name = AsString(interp->StackPop());
        auto coeffs = AsArray(interp->StackPop());

        interp->StackPush(LPEquationItem::New(coeffs, name));
    }
};



// ( varnames objective constraints  -- LinearProgram )
class W_LPNew : public Word
{
public:
    W_LPNew(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        interp->StackPush(shared_ptr<LPItem>(new LPItem(interp)));
    }

};


// ( LinearProgram -- )
class W_LPPrintMatrix : public Word
{
public:
    W_LPPrintMatrix(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        auto linear_program = AsLPItem(interp->StackPop());
        linear_program->PrintMatrix();
    }

};


// ( LinearProgram -- )
class W_LPFree : public Word
{
public:
    W_LPFree(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        auto linear_program = AsLPItem(interp->StackPop());
        linear_program->Free();
    }
};


// =============================================================================
// LinearProgramModule

LinearProgramModule::LinearProgramModule() : Module("linear-program") {
    AddWord(new W_LPNew("LP-NEW"));
    AddWord(new W_LPFree("LP-FREE"));
    AddWord(new W_LPPrintMatrix("LP-PRINT-MATRIX"));
    AddWord(new W_LPEqn("LP-EQN"));
}

string LinearProgramModule::ForthicCode() {
    string result("[ gauss ] USE-MODULES");
    return result;
}

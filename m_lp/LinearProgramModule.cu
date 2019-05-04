#include <ctime>
#include <cstdio>
#include <cmath>
#include "../Interpreter.h"
#include "../m_cuda/CudaModule.h"
#include "LinearProgramModule.h"
#include "../IntItem.h"
#include "../m_cuda/Dim3Item.h"
#include "../ArrayItem.h"
#include "../AddressItem.h"
#include "LPEquationItem.h"
#include "LPItem.h"


// =============================================================================
// Kernels



// =============================================================================
// Words


// ( coeffs name  -- LPEquationItem )
class LPEqnWord : public Word
{
public:
    LPEqnWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        string name = AsString(interp->StackPop());
        auto coeffs = AsArray(interp->StackPop());

        interp->StackPush(LPEquationItem::New(coeffs, name));
    }
};



// ( varnames objective constraints  -- LinearProgram )
class LPNewWord : public Word
{
public:
    LPNewWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        interp->StackPush(shared_ptr<LPItem>(new LPItem(interp)));
    }

};


// ( LinearProgram -- )
class LPPrintMatrixWord : public Word
{
public:
    LPPrintMatrixWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        auto linear_program = AsLPItem(interp->StackPop());
        linear_program->PrintMatrix();
    }

};


// ( LinearProgram -- )
class LPFreeWord : public Word
{
public:
    LPFreeWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        auto linear_program = AsLPItem(interp->StackPop());
        linear_program->Free();
    }
};


// =============================================================================
// LinearProgramModule

LinearProgramModule::LinearProgramModule() : Module("linear-program") {
    AddWord(new LPNewWord("LP-NEW"));
    AddWord(new LPFreeWord("LP-FREE"));
    AddWord(new LPPrintMatrixWord("LP-PRINT-MATRIX"));
    AddWord(new LPEqnWord("LP-EQN"));
}

string LinearProgramModule::ForthicCode() {
    string result("[ gauss ] USE-MODULES");
    return result;
}

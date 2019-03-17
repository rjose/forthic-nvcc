#include <cstdio>
#include "Interpreter.h"
#include "CudaModule.h"
#include "Ch2Module.h"
#include "IntItem.h"
#include "Dim3Item.h"


// =============================================================================
// Kernels
/*
__global__ void helloFromGPU() {
    printf("Hello from GPU!\n");
}
*/




// =============================================================================
// Words


// ( hostref gpuref num -- int )
class CheckResultWord : public Word
{
public:
    CheckResultWord(string name) : Word(name) {};

    virtual void Execute(Interpreter *interp) {
        int num = AsInt(interp->StackPop());
        float* gpuRef = AsFloatStar(interp->StackPop());
        float* hostRef = AsFloatStar(interp->StackPop());

        double epsilon = 1.0E-8;
        bool match = 1;

        for (int i = 0; i < num; i++) {
            if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
                match = 0;
                printf("Arrays do not match!\n");
                printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                       gpuRef[i], i);
                break;
            }
        }
        interp->StackPush(shared_ptr<IntItem>(new IntItem(match)));
    }
};


// =============================================================================
// Ch2Module

Ch2Module::Ch2Module() : Module("ch2") {
    AddWord(shared_ptr<Word>(new CheckResultWord("CHECK-RESULT")));
}

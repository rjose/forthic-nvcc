#include <cstdio>
#include "Interpreter.h"
#include "CudaModule.h"

// =============================================================================
// Words

// ( -- )
// Pops word from stack
class HelloWord : public Word
{
public:
    HelloWord(string name) : Word(name) {};
    virtual void Execute(Interpreter *interp) {
        printf("Hello!\n");
    }
};


CudaModule::CudaModule() : Module("cuda")
{
    AddWord(shared_ptr<Word>(new HelloWord("HELLO")));
}


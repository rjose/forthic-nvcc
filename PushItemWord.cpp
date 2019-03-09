#include "PushItemWord.h"
#include "Interpreter.h"


PushItemWord::PushItemWord(string word_name, shared_ptr<StackItem> i) : Word(word_name), item(i)
{
}


void PushItemWord::Execute(Interpreter *interp)
{
    interp->StackPush(item);
}

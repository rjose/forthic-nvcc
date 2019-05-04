#include "W_Definition.h"
#include "Interpreter.h"


W_Definition::W_Definition(string word_name) : Word(word_name)
{
}

W_Definition::~W_Definition()
{
}

void W_Definition::CompileWord(shared_ptr<Word> word)
{
    words.push_back(word);
}

void W_Definition::Execute(Interpreter *interp)
{
    for (auto iter = words.begin(); iter != words.end(); iter++)
    {
        (*iter)->Execute(interp);
    }
}

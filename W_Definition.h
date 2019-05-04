#pragma once
#include <memory>
#include <string>
#include <vector>
#include "StackItem.h"
#include "Word.h"

using namespace std;

class Interpreter;

class W_Definition : public Word
{
public:
    W_Definition(string name);
    virtual ~W_Definition();
    virtual void Execute(Interpreter *interp);

    void CompileWord(shared_ptr<Word> word);

protected:
    vector<shared_ptr<Word>> words;
};


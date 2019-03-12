#pragma once
#include <memory>
#include <string>
#include <vector>
#include "StackItem.h"
#include "Word.h"

using namespace std;

class Interpreter;

class DefinitionWord : public Word
{
public:
    DefinitionWord(string name);
    virtual ~DefinitionWord();
    virtual void Execute(Interpreter *interp);

    void CompileWord(shared_ptr<Word> word);

protected:
    vector<shared_ptr<Word>> words;
};


#pragma once
#include <string>
#include "StackItem.h"
#include "Word.h"

using namespace std;

class Interpreter;

class EndArrayWord : public Word
{
public:
	EndArrayWord(string name);
	virtual ~EndArrayWord();
	virtual void Execute(Interpreter *interp);
};


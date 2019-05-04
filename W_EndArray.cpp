#include <algorithm>
#include "W_EndArray.h"
#include "Interpreter.h"
#include "./m_global/ArrayItem.h"
#include "StartArrayItem.h"


W_EndArray::W_EndArray(string word_name) : Word(word_name)
{
}


W_EndArray::~W_EndArray()
{
}

void W_EndArray::Execute(Interpreter *interp)
{
	vector<shared_ptr<StackItem>> result;

	while (true)
	{
		auto item = interp->StackPop();
		if (dynamic_cast<StartArrayItem*>(item.get())) break;
		else result.push_back(item);
	}

	std::reverse(result.begin(), result.end());
	interp->StackPush(shared_ptr<StackItem>(new ArrayItem(result)));
}

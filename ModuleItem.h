#pragma once
#include <memory>
#include <string>

#include "GlobalItemGetters.h"
#include "StackItem.h"
#include "Module.h"

using namespace std;


class ModuleItem : public StackItem, public IGetModule
{
public:
	ModuleItem(Module* mod);
	ModuleItem(shared_ptr<Module> mod);
	virtual ~ModuleItem();
	shared_ptr<Module> GetModule();

protected:
	shared_ptr<Module> mod;
};

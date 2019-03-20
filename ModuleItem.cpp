#include "ModuleItem.h"


shared_ptr<Module> ModuleItem::AsModule() {
    return mod;
}


string ModuleItem::AsString() {
    return "ModuleItem";
}

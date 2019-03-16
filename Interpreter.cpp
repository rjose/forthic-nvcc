#include <sstream>
#include "Interpreter.h"
#include "Tokenizer.h"
#include "StringItem.h"
#include "PushItemWord.h"
#include "StartArrayItem.h"
#include "EndArrayWord.h"
#include "ModuleItem.h"

Interpreter::Interpreter() : is_compiling(false)
{
    // The first module in the module_stack is the initial local module
    module_stack.push_back(shared_ptr<Module>(new Module("")));
}


Interpreter::~Interpreter()
{
}


shared_ptr<Module> Interpreter::CurModule()
{
    return module_stack.back();
}


void Interpreter::Run(string input)
{
    Tokenizer tokenizer(input);
    Token tok = tokenizer.NextToken();
    while (tok.GetType() != TokenType::EOS)
    {
        handle_token(tok);
        tok = tokenizer.NextToken();
    }
}


void Interpreter::StackPush(shared_ptr<StackItem> item)
{
    param_stack.push(item);
}

shared_ptr<StackItem> Interpreter::StackPop()
{
    shared_ptr<StackItem> result = param_stack.top();
    param_stack.pop();
    return result;
}


int Interpreter::StackSize() {
    return param_stack.size();
}

void Interpreter::RegisterModule(shared_ptr<Module> mod)
{
    registered_modules[mod->GetName()] = mod;
    this->Run(mod->ForthicCode());
}


void Interpreter::handle_token(Token token)
{
    switch (token.GetType())
    {
    case TokenType::START_ARRAY:
        handle_START_ARRAY(token);
        break;

    case TokenType::END_ARRAY:
        handle_END_ARRAY(token);
        break;

    case TokenType::STRING:
        handle_STRING(token);
        break;

    case TokenType::START_MODULE:
        handle_START_MODULE(token);
        break;

    case TokenType::END_MODULE:
        handle_END_MODULE(token);
        break;

    case TokenType::START_DEFINITION:
        handle_START_DEFINITION(token);
        break;

    case TokenType::END_DEFINITION:
        handle_END_DEFINITION(token);
        break;

    case TokenType::WORD:
        handle_WORD(token);
        break;

    case TokenType::COMMENT:
        break;

    default:
        ostringstream message;
        message << "Unhandled token type: " << (int)(token.GetType());
        throw message.str();
    }
}

void Interpreter::handle_STRING(Token tok)
{
    StringItem* item = new StringItem(tok.GetText());
    auto word = shared_ptr<Word>(new PushItemWord("<string>", shared_ptr<StackItem>(item)));
    handle_Word(word);
}


void Interpreter::handle_Word(shared_ptr<Word> word)
{
    if (is_compiling)  cur_definition->CompileWord(word);
    else word->Execute(this);
}


void Interpreter::handle_START_ARRAY(Token token)
{
    StartArrayItem* item = new StartArrayItem();
    auto word = shared_ptr<Word>(new PushItemWord("[", shared_ptr<StackItem>(item)));
    handle_Word(word);
}


void Interpreter::handle_END_ARRAY(Token token)
{
    auto word = shared_ptr<Word>(new EndArrayWord("]"));
    handle_Word(word);
}


void Interpreter::handle_START_MODULE(Token tok)
{
    // If module has been registered, push it onto the module stack
    if (auto mod = find_module(tok.GetText()))  module_stack_push(mod);

    // Else if the module has no name, push an anonymous module
    else if (tok.GetText() == "")  module_stack_push(shared_ptr<Module>(new Module("")));

    // Else, register a new module under the specified name and push it onto the module stack
    else
    {
        mod = shared_ptr<Module>(new Module(tok.GetText()));
        RegisterModule(mod);
        module_stack_push(mod);
    }
}


void Interpreter::handle_END_MODULE(Token tok)
{
    module_stack.pop_back();
}


shared_ptr<Module> Interpreter::find_module(string name)
{
    if (registered_modules.find(name) == registered_modules.end()) return nullptr;
    else return registered_modules[name];
}

void Interpreter::module_stack_push(shared_ptr<Module> mod)
{
    module_stack.push_back(mod);
}


void Interpreter::handle_START_DEFINITION(Token tok)
{
    if (is_compiling) throw "Can't have nested definitions";
    cur_definition = shared_ptr<DefinitionWord>(new DefinitionWord(tok.GetText()));
    is_compiling = true;
}


void Interpreter::handle_END_DEFINITION(Token tok)
{
    if (!is_compiling) throw "Unmatched end definition";
    CurModule()->AddWord(cur_definition);
    is_compiling = false;
}


void Interpreter::handle_WORD(Token tok)
{
    shared_ptr<Word> word = find_word(tok.GetText());
    if (word == nullptr) throw (string("Unknown word: ") + tok.GetText());
    handle_Word(word);
}

shared_ptr<Word> Interpreter::find_word(string name)
{
    shared_ptr<Word> result = nullptr;

    // Search module stack
    for (auto iter = module_stack.rbegin(); iter != module_stack.rend(); iter++)
    {
        result = (*iter)->FindWord(name);
        if (result != nullptr) break;
    }

    // Treat as registered module
    if (result == nullptr)   result = find_registered_module_word(name);

    // Check global module
    if (result == nullptr)   result = global_module.FindWord(name);

    return result;
}


shared_ptr<Word> Interpreter::find_registered_module_word(string name)
{
    auto mod = find_module(name);
    if (mod == nullptr)  return nullptr;
    else  return shared_ptr<Word>(new PushItemWord(mod->GetName(), shared_ptr<ModuleItem>(new ModuleItem(mod))));
}

ArrayItem.o EndArrayWord.o EndArrayWord.o Interpreter.o StartArrayItem.o  : ArrayItem.h
BasicItemGetters.o test/GlobalModuleTest.o ArrayItem.o ModuleItem.o StringItem.o VariableItem.o : BasicItemGetters.h
CudaModule.o main.o  : CudaModule.h
DefinitionWord.o Interpreter.o : DefinitionWord.h
EndArrayWord.o Interpreter.o  : EndArrayWord.h
FloatItem.o GlobalModule.o  : FloatItem.h
GlobalModule.o test/GlobalModuleTest.o FloatItem.o IntItem.o Interpreter.o : GlobalModule.h
CudaModule.o DefinitionWord.o EndArrayWord.o GlobalModule.o Interpreter.o PushItemWord.o main.o test/GlobalModuleTest.o test/InterpreterTest.o  : Interpreter.h
GlobalModule.o IntItem.o  : IntItem.h
CudaModule.o GlobalModule.o Module.o main.o main.o test/GlobalModuleTest.o test/ModuleTest.o CudaModule.o FloatItem.o GlobalModule.o IntItem.o Interpreter.o Interpreter.o ModuleItem.o : Module.h
Interpreter.o ModuleItem.o  : ModuleItem.h
GlobalModule.o Interpreter.o Module.o PushItemWord.o  : PushItemWord.h
StackItem.o test/InterpreterTest.o ArrayItem.o BasicItemGetters.o DefinitionWord.o EndArrayWord.o FloatItem.o IntItem.o Interpreter.o ModuleItem.o PushItemWord.o StartArrayItem.o StringItem.o VariableItem.o : StackItem.h
EndArrayWord.o Interpreter.o StartArrayItem.o  : StartArrayItem.h
Interpreter.o StringItem.o  : StringItem.h
test/GlobalModuleTest.o test/main_test.o  : test/GlobalModuleTest.h
test/InterpreterTest.o test/main_test.o  : test/InterpreterTest.h
test/GlobalModuleTest.o test/ModuleTest.o test/main_test.o test/main_test.o  : test/ModuleTest.h
test/GlobalModuleTest.o test/InterpreterTest.o test/ModuleTest.o test/Test.o test/TokenizerTest.o test/main_test.o test/main_test.o test/main_test.o test/main_test.o test/GlobalModuleTest.o test/InterpreterTest.o test/ModuleTest.o : test/Test.h
test/TokenizerTest.o test/main_test.o  : test/TokenizerTest.h
Token.o Interpreter.o Tokenizer.o : Token.h
Interpreter.o Tokenizer.o test/TokenizerTest.o  : Tokenizer.h
VariableItem.o Module.o : VariableItem.h
DefinitionWord.o EndArrayWord.o GlobalModule.o Interpreter.o Interpreter.o Module.o PushItemWord.o Word.o DefinitionWord.o EndArrayWord.o Interpreter.o Interpreter.o Module.o PushItemWord.o : Word.h

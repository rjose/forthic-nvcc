LIB_OBJECTS       = Token.o Tokenizer.o Module.o Word.o StackItem.o \
                    BasicItemGetters.o VariableItem.o PushItemWord.o \
                    StringItem.o StartArrayItem.o EndArrayWord.o \
                    GlobalModule.o IntItem.o FloatItem.o \
                    ArrayItem.o DefinitionWord.o ModuleItem.o Interpreter.o
APP_OBJECTS       = main.o $(LIB_OBJECTS)
TEST_OBJECTS      = ./test/Test.o ./test/TokenizerTest.o ./test/ModuleTest.o \
                    ./test/InterpreterTest.o ./test/GlobalModuleTest.o
TEST_APP_OBJECTS  = ./test/main_test.o $(TEST_OBJECTS) $(LIB_OBJECTS)

all: app test runtest

app: $(APP_OBJECTS)
	nvcc -o app $(APP_OBJECTS)

.PHONY: runtest
runtest:
	./test/test

test: $(TEST_APP_OBJECTS)
	nvcc -o ./test/test $(TEST_APP_OBJECTS)

.PHONY: clean
clean:
	rm -f $(APP_OBJECTS) app
	rm -f $(TEST_APP_OBJECTS) ./test/test

%.o:%.cpp %.h
	nvcc -std=c++11 -g -c -o $@ $<

# Dependencies (generate with python3 dep.py)
ArrayItem.o EndArrayWord.o EndArrayWord.o Interpreter.o StartArrayItem.o : ArrayItem.h
BasicItemGetters.o : BasicItemGetters.h
DefinitionWord.o : DefinitionWord.h
EndArrayWord.o Interpreter.o : EndArrayWord.h
FloatItem.o GlobalModule.o : FloatItem.h
GlobalModule.o : GlobalModule.h
DefinitionWord.o EndArrayWord.o GlobalModule.o Interpreter.o PushItemWord.o : Interpreter.h
GlobalModule.o IntItem.o : IntItem.h
GlobalModule.o Module.o : Module.h
Interpreter.o ModuleItem.o : ModuleItem.h
GlobalModule.o Interpreter.o Module.o PushItemWord.o : PushItemWord.h
StackItem.o : StackItem.h
EndArrayWord.o Interpreter.o StartArrayItem.o : StartArrayItem.h
Interpreter.o StringItem.o : StringItem.h
Token.o : Token.h
Interpreter.o Tokenizer.o : Tokenizer.h
VariableItem.o : VariableItem.h
DefinitionWord.o EndArrayWord.o GlobalModule.o Interpreter.o Interpreter.o Module.o PushItemWord.o Word.o : Word.h

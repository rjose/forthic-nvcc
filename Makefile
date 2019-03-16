LIB_OBJECTS       = Token.o Tokenizer.o Module.o Word.o StackItem.o \
                    BasicItemGetters.o VariableItem.o PushItemWord.o \
                    StringItem.o StartArrayItem.o EndArrayWord.o \
                    GlobalModule.o IntItem.o FloatItem.o \
                    CudaModule.o \
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

.PHONY: runapp
runapp: app
	./app

test: $(TEST_APP_OBJECTS)
	nvcc -o ./test/test $(TEST_APP_OBJECTS)

.PHONY: clean
clean:
	rm -f $(APP_OBJECTS) app
	rm -f $(TEST_APP_OBJECTS) ./test/test

%.o:%.cpp %.h
	nvcc -std=c++11 -g -c -o $@ $<

%.o:%.cu %.h
	nvcc -arch=sm_30 -std=c++11 -g -c -o $@ $<

main.o:main.cpp
	nvcc -std=c++11 -g -c -o $@ $<

.PHONY: deps
deps:
	python3 deps.py > deps.mk

# Dependencies (generate with python3 dep.py)
include deps.mk

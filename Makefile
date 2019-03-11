LIB_OBJECTS       = Token.o Tokenizer.o Module.o Word.o StackItem.o \
                    GlobalItemGetters.o VariableItem.o PushItemWord.o \
                    StringItem.o StartArrayItem.o EndArrayWord.o \
					ArrayItem.o Interpreter.o
APP_OBJECTS       = main.o $(LIB_OBJECTS)
TEST_OBJECTS      = ./test/Test.o ./test/TokenizerTest.o ./test/ModuleTest.o \
                    ./test/InterpreterTest.o
TEST_APP_OBJECTS  = ./test/main_test.o $(TEST_OBJECTS) $(LIB_OBJECTS)

all: clean app test

app: $(APP_OBJECTS)
	nvcc -o app $(APP_OBJECTS)

test: $(TEST_APP_OBJECTS)
	nvcc -o ./test/test $(TEST_APP_OBJECTS)
	./test/test

.PHONY: clean
clean:
	rm -f $(APP_OBJECTS) app
	rm -f $(TEST_APP_OBJECTS) ./test/test

%.o:%.cpp
	nvcc -std=c++11 -g -c -o $@ $<

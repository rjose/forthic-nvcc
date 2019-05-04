LIB_OBJECTS       = Token.o Tokenizer.o Module.o Word.o StackItem.o \
                    VariableItem.o PushItemWord.o \
                    StringItem.o StartArrayItem.o EndArrayWord.o \
                    DefinitionWord.o Interpreter.o \
                    m_global/BasicConverters.o m_global/IGetAddress.o \
                    m_global/GlobalModule.o m_global/IntItem.o m_global/FloatItem.o \
                    m_global/AddressItem.o m_global/TimePointItem.o \
                    m_global/ArrayItem.o m_global/ModuleItem.o \
                    m_cuda/CudaModule.o m_cuda/Dim3Item.o \
                    m_cuda/CudaDevicePropItem.o m_gauss/GaussModule.o m_lp/LinearProgramModule.o \
                    m_lp/LPEquationItem.o m_lp/LPItem.o \
                    examples/Ch2Module.o
APP_OBJECTS       = examples/main.o $(LIB_OBJECTS)
TEST_OBJECTS      = test/Test.o test/TokenizerTest.o test/ModuleTest.o \
                    test/InterpreterTest.o test/GlobalModuleTest.o
TEST_APP_OBJECTS  = test/main_test.o $(TEST_OBJECTS) $(LIB_OBJECTS)

all: examples/app test runtest

examples/app: $(APP_OBJECTS)
	nvcc -o examples/app $(APP_OBJECTS) -lncurses

.PHONY: runtest
runtest:
	./test/test

.PHONY: runapp
runapp: examples/app
	cd examples && ./app BHM-p.62-LP.forthic

test: $(TEST_APP_OBJECTS)
	nvcc -o ./test/test $(TEST_APP_OBJECTS) -lncurses

.PHONY: clean
clean:
	rm -f $(APP_OBJECTS) app
	rm -f $(TEST_APP_OBJECTS) ./test/test

%.o:%.cpp %.h
	nvcc -std=c++11 -g -c -o $@ $<

%.o:%.cu %.h
	nvcc -arch=sm_30 -std=c++11 -g -c -o $@ $<

examples/main.o:examples/main.cpp
	nvcc -std=c++11 -g -c -o $@ $<

.PHONY: deps
deps:
	python3 deps.py > deps.mk 2>/dev/null

# Dependencies (generate with python3 dep.py)
include deps.mk

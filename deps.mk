 DefinitionWord.o Interpreter.o : DefinitionWord.h
 EndArrayWord.o Interpreter.o  : EndArrayWord.h
 DefinitionWord.o EndArrayWord.o Interpreter.o PushItemWord.o m_global/GlobalModule.o examples/main.o test/GlobalModuleTest.o test/InterpreterTest.o  : Interpreter.h
   : m_cuda/CudaDevicePropItem.h
 examples/main.o m_cuda/Dim3Item.o m_lp/LPEquationItem.o m_lp/LPItem.o : m_cuda/CudaModule.h
   : m_cuda/Dim3Item.h
 examples/main.o  : m_gauss/GaussModule.h
 m_global/AddressItem.o m_global/GlobalModule.o  : m_global/AddressItem.h
 EndArrayWord.o EndArrayWord.o Interpreter.o StartArrayItem.o m_global/ArrayItem.o m_lp/LPEquationItem.o : m_global/ArrayItem.h
 m_global/BasicConverters.o test/GlobalModuleTest.o StringItem.o VariableItem.o m_global/ArrayItem.o m_global/ModuleItem.o : m_global/BasicConverters.h
 m_global/FloatItem.o m_global/GlobalModule.o  : m_global/FloatItem.h
 m_global/GlobalModule.o test/GlobalModuleTest.o Interpreter.o m_global/FloatItem.o m_global/IntItem.o m_global/TimePointItem.o : m_global/GlobalModule.h
 m_global/GlobalModule.o m_global/IGetAddress.o m_global/AddressItem.o : m_global/IGetAddress.h
 m_global/GlobalModule.o m_global/IntItem.o  : m_global/IntItem.h
 Interpreter.o m_global/ModuleItem.o  : m_global/ModuleItem.h
 m_global/GlobalModule.o m_global/TimePointItem.o m_global/TimePointItem.o : m_global/TimePointItem.h
 examples/main.o  : m_lp/LinearProgramModule.h
   : m_lp/LPEquationItem.h
   : m_lp/LPItem.h
 Module.o m_global/GlobalModule.o examples/main.o examples/main.o examples/main.o examples/main.o examples/main.o test/GlobalModuleTest.o test/ModuleTest.o Interpreter.o Interpreter.o m_cuda/CudaModule.o m_cuda/Dim3Item.o m_gauss/GaussModule.o m_global/FloatItem.o m_global/GlobalModule.o m_global/IntItem.o m_global/ModuleItem.o m_global/TimePointItem.o m_lp/LPEquationItem.o m_lp/LPItem.o m_lp/LinearProgramModule.o examples/Ch2Module.o : Module.h
 Interpreter.o Module.o PushItemWord.o m_global/GlobalModule.o  : PushItemWord.h
 StackItem.o test/InterpreterTest.o DefinitionWord.o EndArrayWord.o Interpreter.o PushItemWord.o StartArrayItem.o StringItem.o VariableItem.o m_cuda/CudaDevicePropItem.o m_cuda/Dim3Item.o m_global/AddressItem.o m_global/ArrayItem.o m_global/BasicConverters.o m_global/FloatItem.o m_global/IGetAddress.o m_global/IntItem.o m_global/ModuleItem.o m_global/TimePointItem.o m_lp/LPEquationItem.o m_lp/LPItem.o : StackItem.h
 EndArrayWord.o Interpreter.o StartArrayItem.o  : StartArrayItem.h
 Interpreter.o StringItem.o m_global/GlobalModule.o  : StringItem.h
 test/GlobalModuleTest.o test/main_test.o  : test/GlobalModuleTest.h
 test/InterpreterTest.o test/main_test.o  : test/InterpreterTest.h
 test/GlobalModuleTest.o test/ModuleTest.o test/main_test.o test/main_test.o  : test/ModuleTest.h
 test/GlobalModuleTest.o test/InterpreterTest.o test/ModuleTest.o test/Test.o test/TokenizerTest.o test/main_test.o test/main_test.o test/main_test.o test/main_test.o test/GlobalModuleTest.o test/InterpreterTest.o test/ModuleTest.o : test/Test.h
 test/TokenizerTest.o test/main_test.o  : test/TokenizerTest.h
 Token.o Interpreter.o Tokenizer.o : Token.h
 Interpreter.o Tokenizer.o test/TokenizerTest.o  : Tokenizer.h
 VariableItem.o m_global/BasicConverters.o Module.o : VariableItem.h
 DefinitionWord.o EndArrayWord.o Interpreter.o Interpreter.o Module.o PushItemWord.o Word.o m_global/GlobalModule.o DefinitionWord.o EndArrayWord.o Interpreter.o Interpreter.o Module.o PushItemWord.o : Word.h

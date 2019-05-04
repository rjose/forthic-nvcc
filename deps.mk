 Interpreter.o W_Definition.o W_EndArray.o W_PushItem.o m_global/GlobalModule.o examples/main.o test/GlobalModuleTest.o test/InterpreterTest.o  : Interpreter.h
   : m_cuda/CudaDevicePropItem.h
   : m_cuda/Dim3Item.h
  m_cuda/Dim3Item.o : m_cuda/I_AsDim3.h
   : m_cuda/M_Cuda.h
 examples/main.o  : m_gauss/GaussModule.h
 m_global/AddressItem.o m_global/GlobalModule.o  : m_global/AddressItem.h
 Interpreter.o StartArrayItem.o W_EndArray.o W_EndArray.o m_global/ArrayItem.o m_lp/LPEquationItem.o : m_global/ArrayItem.h
 m_global/FloatItem.o m_global/GlobalModule.o  : m_global/FloatItem.h
 m_global/GlobalModule.o test/GlobalModuleTest.o Interpreter.o m_global/FloatItem.o m_global/IntItem.o m_global/TimePointItem.o : m_global/GlobalModule.h
 m_global/GlobalModule.o m_global/I_AsArray.o test/InterpreterTest.o m_global/ArrayItem.o : m_global/I_AsArray.h
 m_global/I_AsFloatStar.o m_global/AddressItem.o : m_global/I_AsFloatStar.h
 m_global/I_AsIntStar.o m_global/AddressItem.o : m_global/I_AsIntStar.h
 m_global/GlobalModule.o m_global/I_AsModule.o m_global/ModuleItem.o : m_global/I_AsModule.h
 m_global/I_AsString.o test/InterpreterTest.o StringItem.o : m_global/I_AsString.h
 m_global/I_AsVoidStar.o m_global/AddressItem.o : m_global/I_AsVoidStar.h
 m_global/GlobalModule.o m_global/IntItem.o  : m_global/IntItem.h
 Interpreter.o m_global/ModuleItem.o  : m_global/ModuleItem.h
 m_global/GlobalModule.o m_global/TimePointItem.o m_global/TimePointItem.o : m_global/TimePointItem.h
 examples/main.o  : m_lp/LinearProgramModule.h
   : m_lp/LPEquationItem.h
   : m_lp/LPItem.h
 Module.o m_global/GlobalModule.o m_global/GlobalModule.o m_global/I_AsModule.o examples/main.o examples/main.o examples/main.o examples/main.o examples/main.o test/GlobalModuleTest.o test/ModuleTest.o Interpreter.o Interpreter.o m_cuda/M_Cuda.o m_gauss/GaussModule.o m_global/FloatItem.o m_global/GlobalModule.o m_global/IntItem.o m_global/ModuleItem.o m_global/ModuleItem.o m_global/TimePointItem.o m_lp/LPEquationItem.o m_lp/LPItem.o m_lp/LinearProgramModule.o examples/Ch2Module.o : Module.h
 StackItem.o test/InterpreterTest.o Interpreter.o StartArrayItem.o StringItem.o VariableItem.o W_Definition.o W_EndArray.o W_PushItem.o m_cuda/CudaDevicePropItem.o m_cuda/Dim3Item.o m_global/AddressItem.o m_global/ArrayItem.o m_global/FloatItem.o m_global/I_AsArray.o m_global/I_AsFloatStar.o m_global/I_AsIntStar.o m_global/I_AsModule.o m_global/I_AsString.o m_global/I_AsVoidStar.o m_global/IntItem.o m_global/ModuleItem.o m_global/TimePointItem.o m_lp/LPEquationItem.o m_lp/LPItem.o : StackItem.h
 Interpreter.o StartArrayItem.o W_EndArray.o  : StartArrayItem.h
 Interpreter.o StringItem.o m_global/GlobalModule.o  : StringItem.h
 test/GlobalModuleTest.o test/main_test.o  : test/GlobalModuleTest.h
 test/InterpreterTest.o test/main_test.o  : test/InterpreterTest.h
 test/GlobalModuleTest.o test/ModuleTest.o test/main_test.o test/main_test.o  : test/ModuleTest.h
 test/GlobalModuleTest.o test/InterpreterTest.o test/ModuleTest.o test/Test.o test/TokenizerTest.o test/main_test.o test/main_test.o test/main_test.o test/main_test.o test/GlobalModuleTest.o test/InterpreterTest.o test/ModuleTest.o : test/Test.h
 test/TokenizerTest.o test/main_test.o  : test/TokenizerTest.h
 Token.o Interpreter.o Tokenizer.o : Token.h
 Interpreter.o Tokenizer.o test/TokenizerTest.o  : Tokenizer.h
 VariableItem.o Module.o : VariableItem.h
 W_Definition.o Interpreter.o : W_Definition.h
 Interpreter.o W_EndArray.o  : W_EndArray.h
 Word.o Interpreter.o Module.o W_Definition.o W_EndArray.o W_PushItem.o : Word.h
 Interpreter.o Module.o W_PushItem.o m_global/GlobalModule.o  : W_PushItem.h

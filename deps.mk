m_cuda/CudaModule.o m_gauss/GaussModule.o m_lp/LPItem.o m_lp/LinearProgramModule.o   : AddressItem.h
m_lp/LinearProgramModule.o  m_lp/LPEquationItem.o : ArrayItem.h
  ArrayItem.o ModuleItem.o StringItem.o VariableItem.o : BasicConverters.h
Ch2Module.o   : Ch2Module.h
  Interpreter.o : DefinitionWord.h
   : EndArrayWord.h
m_lp/LPItem.o   : FloatItem.h
m_lp/LPEquationItem.o  FloatItem.o IntItem.o Interpreter.o TimePointItem.o : GlobalModule.h
Ch2Module.o m_cuda/CudaModule.o  AddressItem.o : IGetAddress.h
Ch2Module.o m_cuda/CudaModule.o m_gauss/GaussModule.o m_lp/LPItem.o m_lp/LinearProgramModule.o   : Interpreter.h
Ch2Module.o m_cuda/CudaModule.o m_gauss/GaussModule.o m_lp/LPItem.o m_lp/LinearProgramModule.o   : IntItem.h
m_cuda/CudaDevicePropItem.o m_cuda/CudaModule.o   : m_cuda/CudaDevicePropItem.h
Ch2Module.o m_cuda/CudaModule.o m_gauss/GaussModule.o m_lp/LPItem.o m_lp/LinearProgramModule.o  m_cuda/Dim3Item.o m_lp/LPEquationItem.o m_lp/LPItem.o : m_cuda/CudaModule.h
Ch2Module.o m_cuda/CudaModule.o m_cuda/Dim3Item.o m_gauss/GaussModule.o m_lp/LinearProgramModule.o   : m_cuda/Dim3Item.h
m_gauss/GaussModule.o   : m_gauss/GaussModule.h
m_lp/LinearProgramModule.o   : m_lp/LinearProgramModule.h
m_lp/LPEquationItem.o m_lp/LPItem.o m_lp/LPItem.o m_lp/LinearProgramModule.o   : m_lp/LPEquationItem.h
m_lp/LPItem.o m_lp/LinearProgramModule.o   : m_lp/LPItem.h
Ch2Module.o Ch2Module.o m_cuda/CudaModule.o m_gauss/GaussModule.o m_gauss/GaussModule.o m_lp/LPEquationItem.o m_lp/LPItem.o m_lp/LinearProgramModule.o m_lp/LinearProgramModule.o  Ch2Module.o FloatItem.o GlobalModule.o IntItem.o Interpreter.o Interpreter.o ModuleItem.o TimePointItem.o m_cuda/CudaModule.o m_cuda/Dim3Item.o m_gauss/GaussModule.o m_lp/LPEquationItem.o m_lp/LPItem.o m_lp/LinearProgramModule.o : Module.h
   : ModuleItem.h
   : PushItemWord.h
  AddressItem.o ArrayItem.o BasicConverters.o DefinitionWord.o EndArrayWord.o FloatItem.o IGetAddress.o IntItem.o Interpreter.o ModuleItem.o PushItemWord.o StartArrayItem.o StringItem.o TimePointItem.o VariableItem.o m_cuda/CudaDevicePropItem.o m_cuda/Dim3Item.o m_lp/LPEquationItem.o m_lp/LPItem.o : StackItem.h
   : StartArrayItem.h
m_cuda/CudaModule.o   : StringItem.h
   : test/GlobalModuleTest.h
   : test/InterpreterTest.h
   : test/ModuleTest.h
  test/GlobalModuleTest.o test/InterpreterTest.o test/ModuleTest.o : test/Test.h
   : test/TokenizerTest.h
  TimePointItem.o : TimePointItem.h
  Interpreter.o Tokenizer.o : Token.h
   : Tokenizer.h
  Module.o : VariableItem.h
  DefinitionWord.o EndArrayWord.o Interpreter.o Interpreter.o Module.o PushItemWord.o : Word.h

app: main.o
	nvcc -o app $^

%.o:%.cpp
	nvcc -c -o $@ $<

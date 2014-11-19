FLAGS += -std=c++0x
#FLAGS += -std=c++11

FLAGS += -Wall -Wextra -pedantic
FLAGS += -Ofast

FLAGS +=  `pkg-config opencv --cflags --libs`
FLAGS +=  -lboost_system -lboost_program_options -lboost_serialization -lboost_filesystem

assignment1: *.cpp *.h
	$(CXX) -ggdb -o $@ assignment1.cpp skinmodel.cpp $(FLAGS)

.PHONY : clean
clean:
	rm -f assignment1 graph.txt ROC.png score.txt

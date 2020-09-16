############################################################
####	Makefile template for opencv use
############################################################
RM 			= @rm -rfv
CC = g++
CFLAGS = -g -Wall -std=c++11

SRCS = $(wildcard ./*.cpp)

#SRCS = $(wildcard src/**/*.cpp)	# When source files are distributed in sub folders as following use this SRCS 
#/root
# | obj
# | Makefile
# | src/
# | | dir1/
# | |  | 1.cpp
# | | dir2/
# | |  | 2.cpp
# | |  | dir3/
# | |  |  | 3.cpp
# | |  |  | 4.cpp
# | | main.cpp

#PROG = HelloWorld
PROG      = $(notdir $(shell pwd))	## current foldername is target name

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

#	clean
clean:
	#	$(RM) $(BUILD_PATH)
	$(RM) $(PROG)
	@echo "Done."



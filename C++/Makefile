CXX = g++
BRPC_PATH= ~/github/incubator-brpc/
HDRS+=$(BRPC_PATH)/output/include
LIBS+=$(BRPC_PATH)/output/lib
CXXFLAGS = -std=c++11 -pthread -O2 -fPIC -fno-omit-frame-pointer -I $(HDRS) -L $(LIBS) -lprotobuf -lgflags -lbrpc
ALL: infer_server client
client:
	$(CXX) $(CXXFLAGS) client.cpp echo.pb.o -o $@
infer_server:
	$(CXX) $(CXXFLAGS) server.cpp echo.pb.o -o $@

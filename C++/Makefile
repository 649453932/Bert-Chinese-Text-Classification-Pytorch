CXX = g++
PROTOC = protoc
BRPC_PATH= ~/github/incubator-brpc/
HDRS+=$(BRPC_PATH)/output/include
LIBS+=$(BRPC_PATH)/output/lib
CXXFLAGS = -std=c++11 -pthread -O2 -fPIC -fno-omit-frame-pointer -I $(HDRS) -L $(LIBS) -lprotobuf -lgflags -lbrpc
CXXFLAGS += -I ~/local/onnxruntime/include -L ~/local/onnxruntime/lib -lonnxruntime -I ~/local -I ~/local/include -lutf8proc -L ~/local/lib  -I .
BIN = infer_server client
PROTOS = $(wildcard *.proto)
PROTO_OBJS = $(PROTOS:.proto=.pb.o)
MODEL_OBJ = model.o 
TOKEN_OBJ = tokenization.o

ALL: $(BIN)
client: client.cpp $(PROTO_OBJS)
	@echo "> Linking $@"
	$(CXX) $(CXXFLAGS) $^ -o $@

infer_server: server.cpp $(PROTO_OBJS) $(MODEL_OBJ) $(TOKEN_OBJ)
	@echo "> Linking $@"
	$(CXX) $(CXXFLAGS) $^ -o $@

%.pb.cc %.pb.h: %.proto
	@echo "> Generating $@"
	$(PROTOC) --cpp_out=. --proto_path=. $<

%.o: %.cc
	@echo "> Compiling $@"
	$(CXX) -c $(CXXFLAGS) $< -o $@

%.o: %.cpp
	@echo "> Compiling $@"
	$(CXX) -c $(CXXFLAGS) $< -o $@

%.o: util/%.cpp
	@echo "> Compiling $@"
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -rf *.o *.pb.* $(BIN)

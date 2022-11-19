#include <gflags/gflags.h>
#include <butil/logging.h>
#include <brpc/server.h>
#include "infer.pb.h"
#include "util/model.h"

DEFINE_int32(port, 8000, "TCP Port of this server");
DEFINE_string(listen_addr, "", "Server listen address, may be IPV4/IPV6/UDS."
            " If this is set, the flag port will be ignored");
DEFINE_int32(idle_timeout_s, -1, "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");
DEFINE_int32(logoff_ms, 2000, "Maximum duration of server's LOGOFF state "
             "(waiting for client to close connection before server stops)");

namespace guodongxiaren {
class InferServiceImpl : public InferService {
public:
    InferServiceImpl() {}
    virtual ~InferServiceImpl() { delete model; }
    virtual void NewsClassify(google::protobuf::RpcController* cntl_base,
                      const NewsClassifyRequest* request,
                      NewsClassifyResponse* response,
                      google::protobuf::Closure* done) {
        brpc::ClosureGuard done_guard(done);

        brpc::Controller* cntl =
            static_cast<brpc::Controller*>(cntl_base);

        auto result = model->predict(request->title());
        LOG(INFO) << " " << request->title()
                  << " is " << result;

        response->set_result(request->title());
    }

    int Init(const std::string& model_path, const std::string& vocab_path) {
        model = new Model(model_path, vocab_path);
    }
    Model* model = nullptr;
};
} // namespace guodongxiaren

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    brpc::Server server;

    guodongxiaren::InferServiceImpl service_impl;

    const char* vocab_path = "/home/guodong/bert_pretrain/vocab.txt";
    const char* model_path = "/home/guodong/github/Bert-Chinese-Text-Classification-Pytorch/model.onnx";
    service_impl.Init(model_path, vocab_path);

    if (server.AddService(&service_impl, 
                          brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
        LOG(ERROR) << "Fail to add service";
        return -1;
    }

    butil::EndPoint point;
    if (!FLAGS_listen_addr.empty()) {
        if (butil::str2endpoint(FLAGS_listen_addr.c_str(), &point) < 0) {
            LOG(ERROR) << "Invalid listen address:" << FLAGS_listen_addr;
            return -1;
        }
    } else {
        point = butil::EndPoint(butil::IP_ANY, FLAGS_port);
    }
    brpc::ServerOptions options;
    options.idle_timeout_sec = FLAGS_idle_timeout_s;
    if (server.Start(point, &options) != 0) {
        LOG(ERROR) << "Fail to start InferServer";
        return -1;
    }

    server.RunUntilAskedToQuit();
    return 0;
}

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "udm/memory/MemoryMgr.h"

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 0;
    FLAGS_alsologtostderr = 0;
    FLAGS_timestamp_in_logfile_name = 0;

    // Use stable per-severity files and avoid timestamp filename collisions.
    google::SetLogDestination(google::GLOG_INFO, "celeritas.log");
    google::SetLogDestination(google::GLOG_WARNING, "celeritas.log");
    google::SetLogDestination(google::GLOG_ERROR, "celeritas.log");
    google::SetLogDestination(google::GLOG_FATAL, "celeritas.log");

    testing::InitGoogleTest(&argc, argv);
    const int ret = RUN_ALL_TESTS();
    eUTIL::MemoryMgr::getInstance().shutdown();

    google::ShutdownGoogleLogging();
    return ret;
}

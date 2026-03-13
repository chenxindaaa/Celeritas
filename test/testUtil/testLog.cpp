#include <glog/logging.h>
#include <gtest/gtest.h>

namespace {

void LogErrorIfFalse(bool condition, const char* message) {
    if (!condition) {
        LOG(ERROR) << message;
    }
}

}  // namespace

TEST(test_log, log_error_instead_of_assert) {
    const int lhs = 1 + 1;
    const int rhs = 2;
    LogErrorIfFalse(lhs == rhs, "math check failed: lhs != rhs");
    EXPECT_EQ(lhs, rhs);
}

TEST(test_log, explicit_error_log_path) {
    const bool should_fail = false;
    if (!should_fail) {
        LOG(ERROR) << "intentional error log for test coverage";
    }
    EXPECT_FALSE(should_fail);
}

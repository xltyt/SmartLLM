#include <gtest/gtest.h>
#include <math.h>
#include <glog/logging.h>
#include <net_utils.h>
#include <timer.h>
#include <crypt_utils.h>
#include <string_utils.h>
#include <common.h>

TEST(Common, Time) {
  uint32_t t = mycommon::parseTime("2025-01-01 02:02:02");
  ASSERT_NE(t, (uint32_t)-1);
  t = mycommon::parseTime("2025-xx-01 02:02:02");
  ASSERT_EQ(t, (uint32_t)-1);
  t = mycommon::parseTime("2025-13-01 02:02:02");
  ASSERT_EQ(t, (uint32_t)-1);
  t = mycommon::parseTime("2025-02-31 02:02:02");
  ASSERT_EQ(t, (uint32_t)-1);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */

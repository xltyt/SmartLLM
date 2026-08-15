/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2025.05.11 10:54:36
 *
\****************************************************/

#include "flags.h"

DEFINE_bool(debug, true, "run debug mode");
DEFINE_bool(daemonize, false, "");
DEFINE_bool(standalone, false, "");
DEFINE_int32(http_port, 28800, "");
DEFINE_int32(http_monitor_port, 28801, "");
DEFINE_int32(timeout_connect, 1000 * 500, "");
DEFINE_int32(timeout_read, 1000 * 500, "");
DEFINE_bool(ipv6, false, "");

/* vim: set expandtab nu ts=2 sw=2 sts=2: */

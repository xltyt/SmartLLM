/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2025.05.11 10:54:49
 *
\****************************************************/

#ifndef _FLAGS_H__
#define _FLAGS_H__

#include <gflags/gflags.h>

DECLARE_bool(debug);
DECLARE_bool(daemonize);
DECLARE_bool(standalone);
DECLARE_int32(http_port);
DECLARE_int32(http_monitor_port);
DECLARE_int32(timeout_connect);
DECLARE_int32(timeout_read);
DECLARE_bool(ipv6);

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */

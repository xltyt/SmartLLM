#include "system_utils.h"
#include <stdarg.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <string.h>
#include "common.h"

namespace mycommon {
  int mysystem(const char *cmd, char *output, int maxlen) {
    if (NULL == cmd || NULL == output) {
      return -1;
    }
    FILE *stream = NULL;
    stream = popen(cmd, "r");
    if (NULL == stream) {
      return -2;
    }
    int ret = fread(output, sizeof(char), maxlen - 1, stream);
    pclose(stream);
    return ret;
  }
  
  int mysystem(const char *cmd_param, std::string& output) {
    if (NULL == cmd_param) {
      return -1;
    }
    std::string cmd = cmd_param;
    cmd += " 2>&1";
    FILE *stream = popen(cmd.c_str(), "r");
    if (NULL == stream) {
      return -2;
    }
    long bytes;
    char buf[1024] = {0};
    while ((bytes = fread(buf, sizeof(char), sizeof(buf), stream)) > 0) {
      output += std::string(buf, bytes);
      memset(buf, 0, sizeof(buf));
    }
    pclose(stream);
    return 0;
  }
  
  void mysystem_format(const char *format, ...) {
    char cmd[1024] = {0};
    va_list args;
    va_start(args, format);
    vsnprintf(cmd, _countof(cmd) - 1, format, args);
    va_end(args);
    system(cmd);
  }
  
  std::string get_current_dir() {
    char home[1024];
    int rslt = readlink("/proc/self/exe", home, sizeof(home) - 1); 
    if (rslt < 0 || (rslt >= sizeof(home) - 1)) {
      return "/";
    }
    home[rslt] = '\0';
    char *p = strrchr(home, '/');
    if (NULL != p) {
      *p = 0;
    }
    return home;
  }
}

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */

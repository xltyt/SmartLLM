#ifndef COMMON_TIME_H_INCLUDED
#define COMMON_TIME_H_INCLUDED

#include <time.h>
#include <sys/time.h>
#include <stdint.h>
#include <string>

namespace mycommon {
  class timer {
  public:
    timer();
    ~timer();

    void start();
    void stop();
    double elapsed_microsec();
    double elapsed_millisec();
    double elapsed_sec();
    
    int elapsed_microsec_int();
    int elapsed_millisec_int();
    int elapsed_sec_int();

  private:
    bool _stopped;
    timeval _start_count;
    timeval _end_count;
  };
  
  //获取当前时间，单位：秒
  uint64_t getTime();
  
  //获取当前时间，单位：毫秒
  uint64_t getMilliTime();
  
  //获取当前时间，单位：微秒
  uint64_t getMicroTime();
  
  //获取当前时间，单位：纳秒
  uint64_t getNanoTime();
  
  //获取当前日期（精确到天）,线程不安全
  int getToday(char *buf, int size);
  
  //获取当前小时，分钟, 线程不安全
  long getNow(int &nHour, int &nMinute);
  
  //同上线程安全
  long getSafeNow(int &nHour, int &nMinute);
  
  //同上线程安全, 但返回完整年月日时分秒字符串格式
  long getSafeNow(char *buf, int size);
  std::string getSafeNow();
  long getSafeNowName(char *buf, int size);
  std::string getSafeNowName();
  
  long getSafeNowDay(char *buf, int size);
  long getSafeNowDay1(char *buf, int size);
  
  //把时间转成字符串
  int timeToStr(time_t t, char *buf, int size);
  std::string timeToStr(time_t t);
  std::string timeToDayShort(time_t t);

  int getYear();
  int getMonth();
  int getDay();
  
  uint32_t parseDayShortTime(const char *buf);
  uint32_t parseTime(const char *buf);
  uint32_t timestampToDayTimestamp(time_t t);
  uint32_t timestampToHourTimestamp(time_t t);
  uint32_t timestampDay();
  uint32_t timestampHour();
}

#endif

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */

#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include "timer.h"

namespace mycommon {
  timer::timer() {
    _stopped = false;
    _start_count.tv_sec = 0;
    _start_count.tv_usec = 0;
    _end_count.tv_sec = 0;
    _end_count.tv_usec = 0;
  }

  timer::~timer() {
  }

  void timer::start() {
    _stopped = false;
    gettimeofday(&_start_count, NULL);
  }

  void timer::stop() {
    _stopped = true;
    gettimeofday(&_end_count, NULL);
  }

  double timer::elapsed_microsec() {
    if (!_stopped) {
      gettimeofday(&_end_count, NULL);
    }
    double start_time_in_micro_sec = (_start_count.tv_sec * 1000000.0) + _start_count.tv_usec;
    double end_time_in_micro_sec = (_end_count.tv_sec * 1000000.0) + _end_count.tv_usec;
    return end_time_in_micro_sec - start_time_in_micro_sec;
  }

  double timer::elapsed_millisec() {
    return elapsed_microsec() * 0.001;
  }

  double timer::elapsed_sec() {
    return elapsed_microsec() * 0.000001;
  }
  
  int timer::elapsed_microsec_int() {
    return (int)elapsed_microsec();
  }

  int timer::elapsed_millisec_int() {
    return (int)elapsed_millisec();
  }

  int timer::elapsed_sec_int() {
    return (int)elapsed_sec();
  }

  uint64_t getTime() {
    struct timeval t;
    struct timezone tz;
    gettimeofday(&t, &tz);
    //return (static_cast<uint64_t>(t.tv_sec - tz.tz_minuteswest));
    return (static_cast<uint64_t>(t.tv_sec));
  }

  uint64_t getMilliTime() {
    struct timeval t;
    struct timezone tz;
    gettimeofday(&t, &tz);
    //return (static_cast<uint64_t>(t.tv_sec - tz.tz_minuteswest) * 1000ul
    return (static_cast<uint64_t>(t.tv_sec) * 1000ul
        + static_cast<uint64_t>(t.tv_usec) / 1000ul);
  }

  uint64_t getMicroTime() {
    struct timeval t;
    struct timezone tz;
    gettimeofday(&t, &tz);
    //return (static_cast<uint64_t>(t.tv_sec - tz.tz_minuteswest) * 1000000ul
    return (static_cast<uint64_t>(t.tv_sec) * 1000000ul
        + static_cast<uint64_t>(t.tv_usec));
  }
  
  uint64_t getNanoTime() {
    struct timeval t;
    struct timezone tz;
    gettimeofday(&t, &tz);
    return (static_cast<uint64_t>(t.tv_sec) * 1000000000ul
        + static_cast<uint64_t>(t.tv_usec) * 1000ul);
  }
 
  int getToday(char *buf, int size) {
    struct tm *timeptr;
    time_t ltime;
    time (&ltime);
    if ((timeptr = localtime(&ltime)) == NULL) {
      return -1;
    }
    snprintf(buf, size, "%d%02d%02d", 1900 + timeptr->tm_year, 
        1 + timeptr->tm_mon, timeptr->tm_mday);
    return  0;
  }

  long getNow(int &nHour, int &nMinute) {
    struct tm *timeptr;
    time_t ltime;
    time (&ltime);
    if ((timeptr = localtime(&ltime)) == NULL) {
      return -1;
    }
    nHour = timeptr->tm_hour;
    nMinute = timeptr->tm_min;
    return ltime;
  }

  long getSafeNow(int &nHour, int &nMinute) {
    time_t ltime;
    time (&ltime);
    struct tm ptm = { 0 };  
    if (localtime_r(&ltime, &ptm) == NULL) {
      return -1;
    }
    nHour = ptm.tm_hour;
    nMinute = ptm.tm_min;
    return ltime;
  }

  long getSafeNow(char *buf, int size) {
    time_t ltime;
    time (&ltime);
    struct tm ptm = { 0 };  
    if (localtime_r(&ltime, &ptm) == NULL) {
      return -1;
    }
    snprintf(buf, size, "%d-%02d-%02d %02d:%02d:%02d", 1900 + ptm.tm_year,
        1 + ptm.tm_mon, ptm.tm_mday, ptm.tm_hour, ptm.tm_min, ptm.tm_sec);
    return ltime;
  }
  
  std::string getSafeNow() {
    char buf[512] = {0};
    getSafeNow(buf, sizeof(buf) - 1);
    return buf;
  }
  
  long getSafeNowName(char *buf, int size) {
    time_t ltime;
    time (&ltime);
    struct tm ptm = { 0 };  
    if (localtime_r(&ltime, &ptm) == NULL) {
      return -1;
    }
    snprintf(buf, size, "%d_%02d_%02d_%02d_%02d_%02d", 1900 + ptm.tm_year,
        1 + ptm.tm_mon, ptm.tm_mday, ptm.tm_hour, ptm.tm_min, ptm.tm_sec);
    return ltime;
  }
  
  std::string getSafeNowName() {
    char buf[512] = {0};
    getSafeNowName(buf, sizeof(buf) - 1);
    return buf;
  }
  
  long getSafeNowDay(char *buf, int size) {
    time_t ltime;
    time (&ltime);
    struct tm ptm = { 0 };  
    if (localtime_r(&ltime, &ptm) == NULL) {
      return -1;
    }
    snprintf(buf, size, "%d%02d%02d", 1900 + ptm.tm_year,
        1 + ptm.tm_mon, ptm.tm_mday);
    return ltime;
  }
  
  long getSafeNowDay1(char *buf, int size) {
    time_t ltime;
    time (&ltime);
    struct tm ptm = { 0 };  
    if (localtime_r(&ltime, &ptm) == NULL) {
      return -1;
    }
    snprintf(buf, size, "%d-%02d-%02d", 1900 + ptm.tm_year,
        1 + ptm.tm_mon, ptm.tm_mday);
    return ltime;
  }

  int timeToStr(time_t t, char *str, int size) {
    struct tm ptm;
    localtime_r(&t, &ptm);
    return snprintf(str, size, "%04d-%02d-%02d %02d:%02d:%02d",
        ptm.tm_year+1900, ptm.tm_mon+1, ptm.tm_mday,
        ptm.tm_hour, ptm.tm_min, ptm.tm_sec);
  }
  
  std::string timeToStr(time_t t) {
    char tmp[128] = {0};
    timeToStr(t, tmp, sizeof(tmp));
    return std::string(tmp);
  }
  
  std::string timeToDayShort(time_t t) {
    struct tm ptm;
    localtime_r(&t, &ptm);
    char tmp[128] = {0};
    snprintf(tmp, sizeof(tmp), "%04d-%02d-%02d",
        ptm.tm_year+1900, ptm.tm_mon+1, ptm.tm_mday);
    return std::string(tmp);
  }
  
  int getYear() {
    time_t ltime;
    time (&ltime);
    struct tm ptm = { 0 };  
    if (localtime_r(&ltime, &ptm) == NULL) {
      return -1;
    }
    return 1900 + ptm.tm_year;
  }
  
  int getMonth() {
    time_t ltime;
    time (&ltime);
    struct tm ptm = { 0 };  
    if (localtime_r(&ltime, &ptm) == NULL) {
      return -1;
    }
    return 1 + ptm.tm_mon;
  }
  
  int getDay() {
    time_t ltime;
    time (&ltime);
    struct tm ptm = { 0 };  
    if (localtime_r(&ltime, &ptm) == NULL) {
      return -1;
    }
    return ptm.tm_mday;
  }

  uint32_t parseDayShortTime(const char *buf) {
    struct tm tm;
    memset(&tm, 0, sizeof(struct tm));
    strptime(buf, "%Y-%m-%d", &tm);
    return mktime(&tm);
  }

  uint32_t parseTime(const char *buf) {
    struct tm tm;
    memset(&tm, 0, sizeof(struct tm));
    char *p = strptime(buf, "%Y-%m-%d %H:%M:%S", &tm);
    if (NULL == p) {
      return (uint32_t)-1;
    }
    return (uint32_t)mktime(&tm);
  }

  uint32_t timestampToDayTimestamp(time_t t) {
    struct tm ptm;
    localtime_r(&t, &ptm);
    char buf[128] = {0};
    snprintf(buf, sizeof(buf) - 1, "%04d-%02d-%02d 00:00:00",
      ptm.tm_year + 1900, ptm.tm_mon + 1, ptm.tm_mday);
    return parseTime(buf);
  }
  
  uint32_t timestampToHourTimestamp(time_t t) {
    struct tm ptm;
    localtime_r(&t, &ptm);
    char buf[128] = {0};
    snprintf(buf, sizeof(buf) - 1, "%04d-%02d-%02d %02d:00:00",
      ptm.tm_year + 1900, ptm.tm_mon + 1, ptm.tm_mday,
      ptm.tm_hour);
    return parseTime(buf);
  }
  
  uint32_t timestampDay() {
    time_t t;
    time(&t);
    return timestampToDayTimestamp(t);
  }
  
  uint32_t timestampHour() {
    time_t t;
    time(&t);
    return timestampToHourTimestamp(t);
  }
}

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */

#ifndef COMMON_LOCK_H_INCLUDED
#define COMMON_LOCK_H_INCLUDED

#include <pthread.h>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <string.h>
#include <errno.h>

#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(TypeName)        \
    TypeName(const TypeName&);                    \
    void operator=(const TypeName&)
#endif

namespace common {
  inline static void pthread_call(const char* label, int result) {
    if (result != 0) {
      fprintf(stderr, "pthread %s: %s\n", label, strerror(result));
      abort();
    }
  }

  // pthread mutex wrapper
  class Mutex {
  public:
    Mutex() {
      pthread_call("pthread_mutex_init", pthread_mutex_init(&mutex_, NULL));
    }
    ~Mutex() {
      pthread_call("pthread_mutex_destroy", pthread_mutex_destroy(&mutex_));
    }
    void Lock() {
      pthread_call("pthread_mutex_lock", pthread_mutex_lock(&mutex_));
    }
    void Unlock() {
      pthread_call("pthread_mutex_unlock", pthread_mutex_unlock(&mutex_));
    }

  private:
    friend class CondVar;
    pthread_mutex_t mutex_;
    DISALLOW_COPY_AND_ASSIGN(Mutex);
  };

  // single-threaded placeholder
  class NullMutex {
  public:
    NullMutex() {
    }
    ~NullMutex() {
    }
    void Lock() {
    }
    void Unlock() {
    }
  private:
    DISALLOW_COPY_AND_ASSIGN(NullMutex);
  };

  class RWLock {
  public:
    RWLock() {
      pthread_call("pthread_rwlock_init", pthread_rwlock_init(&rw_lock, NULL));
    }
    ~RWLock() {
      pthread_call("pthread_rwlock_destroy", pthread_rwlock_destroy(&rw_lock));
    }
    void LockR() {
      pthread_call("pthread_rwlock_rdlock", pthread_rwlock_rdlock(&rw_lock));
    }
    void LockW() {
      pthread_call("pthread_rwlock_wrlock", pthread_rwlock_wrlock(&rw_lock));
    }
    void Unlock() {
      pthread_call("pthread_rwlock_unlock", pthread_rwlock_unlock(&rw_lock));
    }
  private:
    pthread_rwlock_t rw_lock;
    DISALLOW_COPY_AND_ASSIGN(RWLock);
  };

  // single-threaded placeholder
  class NullRWLock {
  public:
    void LockR() {
    }
    void LockW() {
    }
    void Unlock() {
    }
  private:
  };

  // single-threaded placeholder
  class MutexSimulativeRWLock {
  public:
    MutexSimulativeRWLock() {
      pthread_call("pthread_mutex_init", pthread_mutex_init(&mutex_, NULL));
    }
    ~MutexSimulativeRWLock() {
      pthread_call("pthread_mutex_destroy", pthread_mutex_destroy(&mutex_));
    }
    void LockR() {
      pthread_call("pthread_mutex_lock", pthread_mutex_lock(&mutex_));
    }
    void LockW() {
      pthread_call("pthread_mutex_lock", pthread_mutex_lock(&mutex_));
    }
    void Unlock() {
      pthread_call("pthread_mutex_unlock", pthread_mutex_unlock(&mutex_));
    }
  private:
    pthread_mutex_t mutex_;
    DISALLOW_COPY_AND_ASSIGN(MutexSimulativeRWLock);
  };

  // condition variable
  class CondVar {
  public:
    explicit CondVar(Mutex* mutex): mutex_(mutex) {
      pthread_call("pthread_cond_init", pthread_cond_init(&cond_var_, NULL));
    }

    ~CondVar() {
      pthread_call("pthread_cond_destroy", pthread_cond_destroy(&cond_var_));
    }

    void Wait() {
      pthread_call("pthread_cond_wait", pthread_cond_wait(&cond_var_, &mutex_->mutex_));
    }

    void Signal() {
      pthread_call("pthread_cond_signal", pthread_cond_signal(&cond_var_));
    }

    void SignalAll() {
      pthread_call("pthread_cond_broadcast", pthread_cond_broadcast(&cond_var_));
    }

  private:
    pthread_cond_t cond_var_;
    Mutex* mutex_;
    DISALLOW_COPY_AND_ASSIGN(CondVar);
  };

  // scoped lock
  template <typename Mutex>
  class ScopedLock
  {
  public:
    ScopedLock(Mutex& mutex): mutex_(&mutex), locked_(false) {
      mutex_->Lock();
      locked_ = true;
    }

    ~ScopedLock() {
      if (locked_) {
        mutex_->Unlock();
      }
    }

    void Lock() {
      mutex_->Lock();
      locked_ = true;
    }

    void UnLock() {
      if (locked_) {
        mutex_->Unlock();
        locked_ = false;
      }
    }

  private:
    Mutex* mutex_;
    bool locked_;
    DISALLOW_COPY_AND_ASSIGN(ScopedLock);
  };
  
  // scoped lock Read
  template <typename Mutex>
  class ScopedLockR
  {
  public:
    ScopedLockR(Mutex& mutex): mutex_(&mutex), locked_(false) {
      mutex_->LockR();
      locked_ = true;
    }

    ~ScopedLockR() {
      if (locked_) {
        mutex_->Unlock();
      }
    }

    void Lock() {
      mutex_->LockR();
      locked_ = true;
    }

    void UnLock() {
      if (locked_) {
        mutex_->Unlock();
        locked_ = false;
      }
    }

  private:
    Mutex* mutex_;
    bool locked_;
    DISALLOW_COPY_AND_ASSIGN(ScopedLockR);
  };
  
  // scoped lock Write
  template <typename Mutex>
  class ScopedLockW
  {
  public:
    ScopedLockW(Mutex& mutex): mutex_(&mutex), locked_(false) {
      mutex_->LockW();
      locked_ = true;
    }

    ~ScopedLockW() {
      if (locked_) {
        mutex_->Unlock();
      }
    }

    void Lock() {
      mutex_->LockW();
      locked_ = true;
    }

    void UnLock() {
      if (locked_) {
        mutex_->Unlock();
        locked_ = false;
      }
    }

  private:
    Mutex* mutex_;
    bool locked_;
    DISALLOW_COPY_AND_ASSIGN(ScopedLockW);
  };

  template <typename T>
  struct SharedInt {
  public:
    SharedInt(): i(0) {
    }
    T Value() {
      ScopedLock<Mutex> scoped_lock(lock);
      return i;
    }
    T Increment(T inc) {
      ScopedLock<Mutex> scoped_lock(lock);
      T ret = i;
      i += inc;
      return ret;
    }
    void Set(T val) {
      ScopedLock<Mutex> scoped_lock(lock);
      i = val;
    }
  private:
    T i;
    Mutex lock;
  };

  class Spin {
  public:
    Spin() {
      pthread_call("pthread_spin_init", pthread_spin_init(&spin_, PTHREAD_PROCESS_SHARED));
    }
    ~Spin() {
      pthread_call("pthread_spin_destroy", pthread_spin_destroy(&spin_));
    }
    void Lock() {
      pthread_call("pthread_spin_lock", pthread_spin_lock(&spin_));
    }
    void Unlock() {
      pthread_call("pthread_spin_unlock", pthread_spin_unlock(&spin_));
    }

  private:
    pthread_spinlock_t spin_;
    DISALLOW_COPY_AND_ASSIGN(Spin);
  };
}

#endif

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */

#include "file_utils.h"
//#include "file_dirent.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <dirent.h>
#include <queue>
#include <algorithm>
#include <string_utils.h>

#include <boost/format.hpp>    
#include <boost/tokenizer.hpp>    
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

namespace mycommon {
  bool file_exist(const std::string& path) {
    struct stat st;
    if (0 != stat(path.c_str(), &st)) {
      return false;
    }

    return true;
  }
  
	long file_size(const std::string& path) {
    FILE *fp = fopen(path.c_str(), "rb");
    if (NULL == fp) {
      return -1;
    }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    fclose(fp);
    return size;
	}

#if 0
  void list_files(std::list<std::string>& list, const std::string& folder, const std::string& extension, bool recursive) {
    Z_DIR* dir;
    Z_DIR* subDir;
    struct z_dirent *ent;
    // try to open top folder
    dir = open_dir(folder.c_str());
    if (dir == NULL){
        // could not open directory
      fprintf(stderr, "Could not open \"%s\" directory.\n", folder.c_str());
      return;
    }else{
        // close, we'll process it next
        close_dir(dir);
    }
    // enqueue top folder
    std::queue<std::string> folders;
    folders.push(folder);

    // run while has queued folders
    while (!folders.empty()){
        std::string currFolder = folders.front();
        folders.pop();
        printf("%s\n", currFolder.c_str());
        dir = open_dir(currFolder.c_str());
        if (dir == NULL) continue;
        // iterate through all the files and directories
        while ((ent = read_dir(dir)) != NULL) {
            std::string name(ent->d_name);
            printf("-%s,%s\n", name.c_str(), ent->d_name);
            // ignore "." and ".." directories
            if ( name.compare(".") == 0 || name.compare("..") == 0) continue;
            // add path to the file name
            std::string path = currFolder;
            path.append("/");
            path.append(name);
            // check if it's a folder by trying to open it
            subDir = open_dir(path.c_str());
            if (subDir != NULL){
                // it's a folder: close, we can process it later
                close_dir(subDir);
                if (recursive) folders.push(path);
            }else{
                // it's a file
                if (extension.empty()){
                    list.push_back(path);
                }else{
                    // check file extension
                    size_t lastDot = name.find_last_of('.');
                    std::string ext = name.substr(lastDot+1);
                    if (ext.compare(extension) == 0){
                      list.push_back(path);
                    }
                } // endif (extension test)
            } // endif (folder test)
        } // endwhile (nextFile)
        close_dir(dir);
    } // endwhile (queued folders
  } // end list_files
#endif

  int unzip(const std::string& src, const std::string& dst) {
    if (!file_exist(dst)) {
      if (-1 == mkdir(dst.c_str(), 0755)) {
        return -1;
      }
    }
    std::string cmd = "unzip " + src + " -d " + dst; 
    if (system(cmd.c_str()) < 0) {
      return -1;
    }
    return 0;
  } // end int unzip

  std::string file_read(const std::string& path) {
    FILE *fp = fopen(path.c_str(), "r");
    if (NULL == fp) {
      return "";
    }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = new (std::nothrow) char[size + 10];
    if (NULL == buf) {
      fclose(fp);
      return "";
    }
    memset(buf, 0, size + 10);
    fread(buf, size, 1, fp);
    fclose(fp);
    std::string str = buf;
    delete[] buf;
    return str;
  }
  
  int file_read(const std::string& path, std::string& content) {
    FILE *fp = fopen(path.c_str(), "r");
    if (NULL == fp) {
      return -1;
    }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = new (std::nothrow) char[size + 10];
    if (NULL == buf) {
      fclose(fp);
      return -1;
    }
    memset(buf, 0, size + 10);
    fread(buf, size, 1, fp);
    fclose(fp);
    content = std::string(buf, size);
    delete[] buf;
    return 0;
  }
  
  void file_read(const std::string& path, std::vector<std::string>& lines) {
    std::string content = file_read(path);
    std::vector<std::string> ori_lines;
    str_split(content, "\n", ori_lines);
    for (auto &line : ori_lines) {
      boost::trim(line);
      if (0 == line.size()) {
        continue;
      }   
      lines.push_back(line);
    }
  }
  
  int file_read_quick(const std::string& path, std::vector<std::string>& lines) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
      return -1;
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      close(fd);
      return -2;
    }
    
    size_t length = sb.st_size;
    char* addr = static_cast<char*>(mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0));
    if (addr == MAP_FAILED) {
      close(fd);
      return -3;
    }
    
    const char* start = addr;
    const char* end = addr + length;
    const char* line_start = addr;
    
    for (const char* p = addr; p < end; ++p) {
      if (*p == '\n') {
        lines.emplace_back(line_start, p - line_start);
        line_start = p + 1;
      }
    }
    
    if (line_start < end) {
      lines.emplace_back(line_start, end - line_start);
    }
    
    munmap(addr, length);
    close(fd);
    return 0;
  }
  
  int file_write(const std::string& path, const std::string& content) {
    FILE *fp = fopen(path.c_str(), "wb");
    if (fp) {
      int ret = fwrite(content.c_str(), 1, content.size(), fp);
      fclose(fp);
      return ret;
    }
    return -1;
  }

  int get_current_dir(std::string& dir) {
    char home[1024];
    int rslt = readlink("/proc/self/exe", home, sizeof(home) - 1);
    if (rslt < 0 || (rslt >= sizeof(home) - 1)) {
      return -1;
    }
    home[rslt] = '\0';
    char *p = strrchr(home, '/');
    if (NULL != p) {
      *p = 0;
    }
    dir = home;
    return 0;
  }
  
  static void EnumFilesInternal(const std::string& dir_str, std::vector<EnumFileInfo>& files, int level, int& cur_level) {
    DIR *dir;
    if ((dir = opendir(dir_str.c_str())) != NULL) {
      struct dirent *de;
      while ((de = readdir(dir)) != NULL) {
        if (!strcmp(de->d_name, ".")) {
          continue;
        }
        if (!strcmp(de->d_name, "..")) {
          continue;
        }
        std::string path = dir_str + "/" + de->d_name;
        std::string name = de->d_name;
        struct stat st;
        if (0 != stat(path.c_str(), &st)) {
          continue;
        }
        bool added = false;
        if (S_ISDIR(st.st_mode)) {
          cur_level++;
          if (cur_level <= level) {
            EnumFilesInternal(path, files, level, cur_level);
            added = true;
          }
          cur_level--;
        }
        if (!added) {
          EnumFileInfo info;
          info.filepath = path;
          info.filename = name;
          info.mtime = st.st_mtime;
          files.push_back(info);
        }
      }
      closedir(dir);
    }
  }
  
  static bool _Less(const EnumFileInfo& lhs, const EnumFileInfo& rhs) {
    return lhs.mtime < rhs.mtime;
  }

  void EnumFiles(const std::string& dir_str, std::vector<std::string>& files, int level /*= 0*/, bool sort_asc_time /*= true*/) {
    std::vector<EnumFileInfo> infos;
    int cur_level = 0;
    EnumFilesInternal(dir_str, infos, level, cur_level);
    if (sort_asc_time) {
      std::sort(infos.begin(), infos.end(), _Less);
    }
    for (std::vector<EnumFileInfo>::const_iterator iter = infos.begin(); iter != infos.end(); iter++) {
      files.push_back(iter->filepath);
    }
  }
  
  void EnumFiles(const std::string& dir_str, std::vector<EnumFileInfo>& files, int level /*= 0*/) {
    int cur_level = 0;
    EnumFilesInternal(dir_str, files, level, cur_level);
  }
  
  int mkdirs(const std::string& dir, std::string& error) {
		int ret = -1;
		try {
      ret = (true == boost::filesystem::create_directories(dir) ? 0 : -1);
    } 
    catch (const boost::filesystem::filesystem_error& e) {
			error = e.what();
    }
		return ret;
  }
  
  int lock_file(const std::string& f, std::function<void()> func) {
    int fd = open(f.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
      perror("open failed");
      return -1;
    }

    if (flock(fd, LOCK_EX) == -1) {
      perror("flock failed");
      close(fd);
      return -2;
    }

    func();

    flock(fd, LOCK_UN);
    return 0;
  }
}

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */

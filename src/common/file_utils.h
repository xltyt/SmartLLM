/****************************************************\
 *
 * Copyright (C) 2017 All Rights Reserved
 * Last modified: 2025.05.07 17:44:01
 *
\****************************************************/


#ifndef _FILE_UTILS_H__
#define _FILE_UTILS_H__

#include <string>
#include <vector>
#include <list>
#include <functional>

namespace mycommon {
  //! Determines if a file exists and could be opened. 
  //! \note The file CAN be a directory
  /** \param filename is the string identifying the file which should be tested for existence.
  \return Returns true if file exists, and false if it does not exist or an error occurred. */
  bool file_exist(const std::string& path);
  long file_size(const std::string& path);

  // \brief List files of a directory.
  // \param list The output files list.
  // \param folder The folder where to search in
  // \param extension A file extension filter,
  // empty extension will match all files.
  // \param recursive If true it will list files in
  // sub directories as well.
  //void list_files(std::list<std::string>& list, const std::string& folder, const std::string& extension = "", bool recursive = false);

  int unzip(const std::string& src, const std::string& dst);

  std::string file_read(const std::string& path);
  int file_read(const std::string& path, std::string& content);
  void file_read(const std::string& path, std::vector<std::string>& lines);
  int file_read_quick(const std::string& path, std::vector<std::string>& lines);
  int file_write(const std::string& path, const std::string& content);

  int get_current_dir(std::string& dir);
  
  void EnumFiles(const std::string& dir_str, std::vector<std::string>& files, int level = 0, bool sort_asc_time = true);
  
  struct EnumFileInfo {
    std::string filepath;
    std::string filename;
    uint32_t mtime;
  };
  void EnumFiles(const std::string& dir_str, std::vector<EnumFileInfo>& files, int level = 0);

  int mkdirs(const std::string& dir, std::string& error);

  int lock_file(const std::string& f, std::function<void()> func);
}

#endif

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */

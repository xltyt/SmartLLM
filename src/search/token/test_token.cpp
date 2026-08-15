/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.08.15 23:31:08
 *
\****************************************************/

#include <iostream>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <iostream>
#include <fstream>
#include <string>
#include <future>
#include <string_utils.h>
#include "core_bpe.h"
#include "qwen_token.h"
#include "common.h"

#if 0
template<typename REQ>
void RunMulti(const std::vector<REQ>& lines, int cpu_count, std::function<void(const REQ& line)> func) {
  std::vector<int> step_count;
  int local_index_total = lines.size() / cpu_count;
  int left = lines.size();
  for (int i = 0; i < (int)cpu_count; i++) {
    step_count.push_back(local_index_total);
    left -= local_index_total;
  }
  int i = 0;
  while (left > 0) {
    step_count[i]++;
    left--;
    i++;
  }
  int index = 0;
  auto job_results = std::vector<std::future<int>>();
  for (int i = 0; i < cpu_count; i++) {
    std::vector<REQ> step_lines;
    for (int j = index; j < index + step_count[i]; j++) {
      step_lines.push_back(lines[j]);
    }   
    index += step_count[i];
    job_results.push_back(std::async(std::launch::async, [step_lines, func]() -> int {
      for (typename std::vector<REQ>::const_iterator iter = step_lines.begin(); iter != step_lines.end(); iter++) {
        func(*iter);
      }   
      return 0;
    }));
  }
  for (auto &result : job_results) {
    result.get();
  }
}

template<typename REQ>
void RunMulti(const std::vector<REQ>& lines, int cpu_count, std::function<void(const REQ& line)> func) {
  if (cpu_count <= 0 || lines.empty()) {
		return;
	}

  size_t total_size = lines.size();
  size_t chunk_size = total_size / cpu_count;
  size_t remainder = total_size % cpu_count;

  std::vector<std::future<int>> job_results;
  job_results.reserve(cpu_count);

  size_t index = 0;
  for (int i = 0; i < cpu_count; ++i) {
    size_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);
    
    auto begin = lines.begin() + index;
    auto end = begin + current_chunk_size;
    
    job_results.push_back(std::async(std::launch::async, 
      [begin, end, &func]() -> int {
        for (auto it = begin; it != end; ++it) {
          func(*it);
        }
        return 0;
      }
    ));
    index += current_chunk_size;
  }

  for (auto& result : job_results) {
    result.get();
  }
}
#endif

template<typename REQ>
void RunMulti(const std::vector<REQ>& lines, int cpu_count, std::function<void(const REQ& line)> func) {
  if (cpu_count <= 0 || lines.empty()) return;

  std::atomic<size_t> next_index{0};
  std::vector<std::future<int>> futures;
  futures.reserve(cpu_count);

  for (int i = 0; i < cpu_count; ++i) {
    futures.push_back(std::async(std::launch::async, [&]() -> int {
      while (true) {
        size_t index = next_index.fetch_add(1);
        if (index >= lines.size()) break;
        func(lines[index]);
      }
      return 0;
    }));
  }

  for (auto& f : futures) {
    f.get();
  }
}

TEST(Token, Utf8) {
  {
    std::vector<uint32_t> ids = CoreBPE::decode_utf8("Ġ");
    ASSERT_EQ(ids.size(), 1);
    ASSERT_EQ(ids[0], 288);
  }
  {
    std::vector<uint32_t> ids = CoreBPE::decode_utf8("Ġt");
    ASSERT_EQ(ids.size(), 2);
    ASSERT_EQ(ids[0], 288);
    ASSERT_EQ(ids[1], 116);
  }
  {
    std::vector<uint32_t> ids = CoreBPE::decode_utf8("Ġbr");
    ASSERT_EQ(ids.size(), 3);
    ASSERT_EQ(ids[0], 288);
    ASSERT_EQ(ids[1], 98);
    ASSERT_EQ(ids[2], 114);
  }
}

TEST(Token, TikTokenEncodingGpt) {
  const std::string& ENDOFTEXT = "<|endoftext|>";
  const std::string& FIM_PREFIX = "<|fim_prefix|>";
  const std::string& FIM_MIDDLE = "<|fim_middle|>";
  const std::string& FIM_SUFFIX = "<|fim_suffix|>";
  const std::string& ENDOFPROMPT = "<|endofprompt|>";

  auto ToString = [](const std::vector<size_t>& ids) -> std::string {
    std::string str;
    for (int i = 0; i < ids.size(); i++) {
      if (str.size()) {
        str += " ";
      }
      str += std::to_string(ids[i]);
    }
    return str;
  };
  // gpt2
  auto mergeable_ranks = CoreBPE::data_gym_to_mergeable_bpe_ranks("/data/gpt2/vocab.bpe", "/data/gpt2/encoder.json");
  LOG(INFO) << "Convert BPE Finished, Len[" << mergeable_ranks.size() << "]";
  ASSERT_EQ(50256, mergeable_ranks.size());
  CoreBPE::StringMap<size_t> special_tokens;
  special_tokens[ENDOFTEXT] = 50256;
  TikTokenEncoding encoding(
    "gpt2",
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
    //"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+",
    //"'s|'t|'re|'ve|'m|'ll|'d|\\s+",
    mergeable_ranks,
    special_tokens,
    50257
    );
  LOG(INFO) << "Init Encoding Finished";
  ASSERT_EQ(true, encoding.encode("hello world") == std::vector<size_t>({31373, 995}));
  ASSERT_EQ(true, encoding.encode("hello <|endoftext|>", std::unordered_set<std::string>{"all"}) == std::vector<size_t>({31373, 220, 50256}));
  ASSERT_EQ(true, encoding.encode("<|endoftext|>", std::unordered_set<std::string>{"all"}) == std::vector<size_t>({50256}));
  ASSERT_EQ(true, encoding.encode("<|endoftext|>", std::unordered_set<std::string>{"<|endoftext|>"}) == std::vector<size_t>({50256}));
  ASSERT_EQ(true, encoding.encode("<|endoftext|>", std::unordered_set<std::string>{}, std::unordered_set<std::string>{}) == std::vector<size_t>({27, 91, 437, 1659, 5239, 91, 29}));
	try {
  	std::vector<size_t> ids = encoding.encode("<|endoftext|>");
    ASSERT_EQ(true, false);
	}
	catch (const std::exception& e) {
    ASSERT_EQ(true, true);
	}
	ASSERT_EQ(true, encoding.encode("0") == std::vector<size_t>({15}));
  ASSERT_EQ(true, encoding.encode("000000000") == std::vector<size_t>({10535, 830}));
  ASSERT_EQ(true, encoding.encode("00000000000000000") == std::vector<size_t>({8269, 10535, 830}));
  ASSERT_EQ(true, encoding.encode("今天天气真不错") == std::vector<size_t>({20015, 232, 25465, 25465, 36365, 242, 40367, 253, 38834, 165, 242, 247}));
  ASSERT_EQ(true, encoding.encode("i'm Jack") == std::vector<size_t>({72, 1101, 3619}));
  ASSERT_EQ(true, encoding.encode("") == std::vector<size_t>());
  LOG(INFO) << "Finished";
}

#if 0
TEST(Token, TikTokenEncodingCl100K) {
  CoreBPE::HashMap<std::vector<uint8_t>, size_t> mergeable_ranks = CoreBPE::load_tiktoken_bpe("/data/cl100k_base/cl100k_base.tiktoken");
  LOG(INFO) << "Convert BPE Finished, Len[" << mergeable_ranks.size() << "]";
  ASSERT_EQ(100256, mergeable_ranks.size());
  CoreBPE::StringMap<size_t> special_tokens;
  special_tokens["ENDOFTEXT"] = 100257;
  special_tokens["FIM_PREFIX"] = 100258;
  special_tokens["FIM_MIDDLE"] = 100259;
  special_tokens["FIM_SUFFIX"] = 100260;
  special_tokens["ENDOFPROMPT"] = 100276;
  TikTokenEncoding encoding(
    "cl100k_base",
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    mergeable_ranks,
    special_tokens,
    50257
    );
  LOG(INFO) << "Init Encoding Finished";
  ASSERT_EQ(true, encoding.encode("rer") == std::vector<size_t>({38149}));
  ASSERT_EQ(true, encoding.encode("'rer") == std::vector<size_t>({2351, 81}));
  ASSERT_EQ(true, encoding.encode("today\n ") == std::vector<size_t>({31213, 198, 220}));
  ASSERT_EQ(true, encoding.encode("today\n \n") == std::vector<size_t>({31213, 27907}));
  ASSERT_EQ(true, encoding.encode("today\n  \n") == std::vector<size_t>({31213, 14211}));
  ASSERT_EQ(true, encoding.encode("hello world") == std::vector<size_t>({15339, 1917}));
  //std::vector<size_t> ids = encoding.encode(" \x85""0");
  //LOG(INFO) << ids.size();
  //for (auto id : ids) {
  //  LOG(INFO) << id;
  //}
  //ASSERT_EQ(true, encoding.encode(" \x85""0") == std::vector<size_t>({220, 126, 227, 15}));
    
  ASSERT_EQ(true, encoding.encode("👍") == std::vector<size_t>({9468, 239, 235}));
  //
  //  # surrogate pair gets converted to codepoint
  //  assert enc.encode("") == []
  //  # lone surrogate just gets replaced
  //  assert enc.encode("\ud83d") == enc.encode("�")
  //ASSERT_EQ(true, encoding.encode("\ud83d\udc4d") == std::vector<size_t>({9468, 239, 235}));
  //ASSERT_EQ(true, encoding.encode("") == std::vector<size_t>({}));
  LOG(INFO) << "Finished";
}
#endif

TEST(Token, TikTokenEncodingR50K) {
}

TEST(Token, Load) {
  // Qwen3
  auto mergeable_ranks = CoreBPE::data_gym_to_mergeable_bpe_ranks("/data/Qwen3-0.6B/tokenizer.json");
  LOG(INFO) << "Convert BPE Finished, Len[" << mergeable_ranks.size() << "]";
}

TEST(Token, QwenEncodeBase) {
  std::vector<uint32_t> utf_data = CoreBPE::decode_utf8("इस एड्रेस बुक से जुड़ा एड्रेस");
  LOG(INFO) << "Utf[" << mycommon::str_join(utf_data) << "]";
  
  QwenToken token("/data/Qwen3-0.6B/");
#if 0
  ASSERT_EQ(true, token.encode("Hello") == std::vector<size_t>({9707}));
  ASSERT_EQ(true, token.encode("中国") == std::vector<size_t>({58695}));
  ASSERT_EQ(true, token.encode("中国<|endoftext|>") == std::vector<size_t>({58695, 151643}));
  ASSERT_EQ(true, token.encode("👍") == std::vector<size_t>({144349}));
  ASSERT_EQ(true, token.encode("पता एडिट करना") == std::vector<size_t>({86162, 79238, 23868, 14925, 237, 146187, 42311, 253, 47809, 44179, 60096, 23868}));
  ASSERT_EQ(true, token.encode("&amp;लेबल") == std::vector<size_t>({27066, 26, 91811, 54784, 105, 91811}));
  {
    std::vector<size_t> ids = token.encode("इस एड्रेस बुक से जुड़ा एड्रेस");
    LOG(INFO) << mycommon::str_join(ids);
    ASSERT_EQ(false, ids == std::vector<size_t>({146575, 78368, 14925, 237, 146187, 85033, 54784, 116, 14925, 105, 72653, 64704, 68158, 34370, 14925, 250, 72653, 146187, 5502, 120, 23868, 14925, 237, 146187, 85033, 54784, 116}));
    ids = token.encode(mycommon::utf8_normalize("इस एड्रेस बुक से जुड़ा एड्रेस"));
    LOG(INFO) << mycommon::str_join(ids);
    ASSERT_EQ(true, ids == std::vector<size_t>({146575, 78368, 14925, 237, 146187, 85033, 54784, 116, 14925, 105, 72653, 64704, 68158, 34370, 14925, 250, 72653, 146187, 5502, 120, 23868, 14925, 237, 146187, 85033, 54784, 116}));
  }
  ASSERT_EQ(true, token.encode("&amp;पता") == std::vector<size_t>({27066, 26, 86162, 79238, 23868}));
//  ASSERT_EQ(true, token.encode("इस एड्रेस बुक से जुड़ी प्रविष्टि केवल भेजने वाले addresses के लिए बदली जा सकती है|") == std::vector<size_t>({146575, 78368, 14925, 237, 146187, 85033, 54784, 116, 14925, 105, 72653, 64704, 68158, 34370, 14925, 250, 72653, 146187, 5502, 120, 43647, 83636, 85033, 145535, 42311, 115, 30484, 253, 38851, 47809, 54784, 113, 91811, 14925, 255, 54784, 250, 60096, 34370, 14925, 113, 31411, 110, 34370, 14230, 47809, 34370, 14925, 110, 42311, 237, 14925, 105, 145256, 91811, 43647, 14925, 250, 23868, 68158, 64704, 79238, 43647, 84310, 12619, 230, 91}));
  ASSERT_EQ(true, token.encode("नया स्वीकार्य पता") == std::vector<size_t>({60096, 145420, 23868, 68158, 30484, 113, 43647, 64704, 31411, 108, 30484, 107, 83636, 79238, 23868}));
  ASSERT_EQ(true, token.encode("नया भेजने वाला पता") == std::vector<size_t>({60096, 145420, 23868, 14925, 255, 54784, 250, 60096, 34370, 14925, 113, 31411, 110, 23868, 83636, 79238, 23868}));
  ASSERT_EQ(true, token.encode("एडिट स्वीकार्य पता ") == std::vector<size_t>({146049, 146187, 42311, 253, 68158, 30484, 113, 43647, 64704, 31411, 108, 30484, 107, 83636, 79238, 23868, 220}));
  {
    LOG(INFO) << "========= Multi Thread Check ===========";
    auto ids = token.encode("EuropeRussiaSt Petersburg\nHistoric Heart activities, tickets and more\n0 hours3 days\nSt. Petersburg Shore Excursion 2-Day City Group Tour with Visa\nThis city tour is the perfect way to explore St. Petersburg in 2 days. During this comprehensive group tour");
    LOG(INFO) << mycommon::str_join(ids);
    ids = token.encode(mycommon::utf8_normalize("EuropeRussiaSt Petersburg\nHistoric Heart activities, tickets and more\n0 hours3 days\nSt. Petersburg Shore Excursion 2-Day City Group Tour with Visa\nThis city tour is the perfect way to explore St. Petersburg in 2 days. During this comprehensive group tour"));
    LOG(INFO) << mycommon::str_join(ids);
    ids = token.encode(mycommon::utf8_normalize("EuropeRussiaSt Petersburg\nHistoric Heart activities, tickets and more\n0 hours3 days\nSt. Petersburg Shore Excursion 2-Day City Group Tour with Visa\nThis city tour is the perfect way to explore St. Petersburg in 2 days. During this comprehensive group tour you will see the major attractions of St. Petersburg including the Hermitage, Peterhof summer residence with the Fountain park, and Catherine Palace with the Amber room. You will pay a visit to the Church on the Spilled Blood which is beautifully decorated with mosaics and enjoy a stroll along the main city's street - Nevsky prospect. The tour won't be complete without a relaxing boat ride along the legendary Neva river. Finally, you'll be able to visit one of the most beautiful subways in the world! During the tour you will hear stories about the city, its foundation, emperors and empresses that changed its appearance through centuries, interesting facts about the Soviet period and learn a lot about modern life in Russia: housing, transportation, health care system, family life and much more.\n\u2022 16 hours\nSt. Petersburg Visa-Free 2-Day Shore Excursion with Boat Ride\nDiscover St. Petersburg with your English-speaking guide on this active and comprehensive two-day excursion. You will visit the world-famous Hermitage Museum, Peterhof Fountain Park and Gardens, Peter and Paul's Fortress and Cathedral, the iconic Church of our Saviour on the Spilled Blood, St. Isaac's Cathedral, Yusupov Palace (the site of Rasputin's murder) and more. Your local tour guide will bring the city's fascinating and grandiose history to life with stories and anecdotes. You will ride on the city's River and canals, enjoy two traditional Russian lunches and immerse yourself in this beautiful city on this small-group shore excursion for ships in port for 2 days. This excursion is Visa-free for cruise ship passengers. Please provide passport information at time of booking so that we may issue the Visa-waiver document prior to your arrival.\n\u2022 2 days\nTour of Pushkin (Tsarskoye Selo) and Catherine Palace\nVisit the magnificent summer residence of the Russian Tsars. A comfortable chauffeured drive will take you to the town of Pushkin (Tsarskoye Selo), where you will find a jewel of Russian baroque- Catherine Palace with the Amber Room inside, which is considered to be one of the world's wonders. Numbers are limited to 9 people on this small-group tour, ensuring you'll receive personalized attention from your guide. For more Russian royal history, upgrade your tour to include Pavlovsk Palace.\n\u2022 5 hours\nSt. Petersburg 2-Day Shore Tour: Hermitage, Catherine Palace\nOn this 2-day cultural tour of St. Petersburg, disembark from your cruise ship and explore the city\u2019s most impressive sights and splendors. Visit the world-famous Hermitage Museum, explore Russian Imperial palaces, venture to the Versailles-rivaling Peterhof Fountain Park, and go for a canal cruise \u2013 it\u2019s no accident that St. Petersburg is also known as the \u201cVenice of the North.\u201d Guides will ensure your tour of the city goes smoothly during this 17-hour excursion. You can also sign on for an additional, 3-hour extension, during which you can experience St. Petersburg's various nighttime attractions, from the theatre to folklore performances.\nSt. Petersburg Hermitage Museum Skip-the-Line Ticket and Tour\nBeat the crowds and lines at St Petersburg\u2019s Hermitage Museum with this 3-hour tour inside one of the world\u2019s greatest art galleries. With an expert guide, enjoy skip-the-line entry, and between May and September, gain early admission so you can start your tour before the crowds. Enjoy personalized attention from your guide on this small-group tour, limited to ten people.\nGrand Tour of St Petersburg\nThe Grand Tour of St Petersburg offers visitors the opportunity to take in as much as possible of the city in a 3-hour sightseeing: a complete tour of St Petersburg's most famous sites from the comfort of a car or mini van and with a knowledgeable and experienced guide. The tour is an excellent introduction to St Petersburg -- the second largest and most beautiful city in Russia. Over almost 300 years of its history St Petersburg accumulated all the grandeur of the Russian Imperial Court and became one of the largest centers of culture and science. Numbers are limited to six people on this small-group tour, ensuring you'll receive personalized attention from your guide.\nSt Petersburg Shore Excursion: Visa-Free 2-day Tour\nThis small-group (max 16 people) visa-free St Petersburg Cruise Excursion will give you an opportunity to see all highlights of St. Petersburg.\nSt. Petersburg Visa-Free 2-Day All Highlights Group Tour\nEnjoy this comprehensive 2-day all-highlights group tour at the lowest price possible, yet with a comfort of sharing a small mini-bus with several other passengers. This is a Visa-Free service which means you do not need to obtain a Russian Visa required. Group size is limited to 16 people.\nSt. Petersburg 2-Day Grand Shore Excursion Tour\nThis small group (maximum 16 people), all-inclusive two day grand tour offers the most dazzling highlights of St. Petersburg for those who wish to make the most of their 2 days in this beautiful and exciting city and is the perfect way to explore St. Petersburg! Explore the best of magnificent St. Petersburg in a small group tour, allowing for a more personalized and intimate experience with premium service including lunches, a canal boat ride. The 2 day grand tour covers all of the most dazzling must-see sights of St. Petersburg with our dedicated, friendly, hand-picked guides who bring to life the fascinating sights and history of the city. This comprehensive tour is ideal for guests who wish to make the most of their 2 day stay!\nSt. Petersburg and Faberge Museum 2-Day Tour\nThis two-day group excursion is a great way to see the most well-known sights of St. Petersburg along with the Faberge museum, which has become one of the new gems in the city. Apart from this museum, you'll visit two royal residences, including the Catherine Palace in Pushkin, and Peterhof with Fountain park. Also, see the inside of the Church on the Spilled Blood and St. Isaac's Cathedral. One of the major attractions you pay a visit to is certainly the Hermitage, the largest museum in the country. During this tour you will learn interesting facts about the present and past history of the city, hear the stories about the Romanov family and gain a deeper understanding of Russian culture, traditions and everyday life. Our guides will make the history of the city come alive and our professional team will do everything to make your time in this beautiful city very special."));
    ASSERT_EQ(true, ids == std::vector<size_t>({30780, 44506, 623, 53948, 198, 48983, 292, 17965, 7488, 11, 14403, 323, 803, 198, 15, 4115, 18, 2849, 198, 623, 13, 53948, 44719, 38895, 34280, 220, 17, 54912, 4311, 5737, 14644, 448, 43334, 198, 1986, 3283, 7216, 374, 279, 4727, 1616, 311, 13186, 794, 13, 53948, 304, 220, 17, 2849, 13, 11954, 419, 15817, 1874, 7216, 498, 686, 1490, 279, 3598, 38491, 315, 794, 13, 53948, 2670, 279, 6252, 1763, 424, 11, 11044, 75858, 7324, 21682, 448, 279, 77224, 6118, 11, 323, 41563, 30296, 448, 279, 46664, 3054, 13, 1446, 686, 2291, 264, 3947, 311, 279, 9257, 389, 279, 3089, 4374, 20070, 892, 374, 31619, 36009, 448, 296, 11983, 1211, 323, 4669, 264, 68883, 3156, 279, 1887, 3283, 594, 8592, 481, 24324, 26684, 21479, 13, 576, 7216, 2765, 944, 387, 4583, 2041, 264, 33848, 15328, 11877, 3156, 279, 27712, 4182, 6586, 14796, 13, 17375, 11, 498, 3278, 387, 2952, 311, 3947, 825, 315, 279, 1429, 6233, 1186, 2284, 304, 279, 1879, 0, 11954, 279, 7216, 498, 686, 6723, 7343, 911, 279, 3283, 11, 1181, 16266, 11, 976, 712, 1087, 323, 976, 1873, 288, 429, 5497, 1181, 11094, 1526, 23631, 11, 7040, 13064, 911, 279, 19390, 4168, 323, 3960, 264, 2696, 911, 6481, 2272, 304, 8359, 25, 11721, 11, 17903, 11, 2820, 2453, 1849, 11, 2997, 2272, 323, 1753, 803, 624, 6667, 220, 16, 21, 4115, 198, 623, 13, 53948, 43334, 62890, 220, 17, 54912, 44719, 38895, 34280, 448, 44232, 40913, 198, 50002, 794, 13, 53948, 448, 697, 6364, 61190, 8474, 389, 419, 4541, 323, 15817, 1378, 11228, 94340, 13, 1446, 686, 3947, 279, 1879, 2220, 22517, 6252, 1763, 424, 16328, 11, 11044, 75858, 77224, 5540, 323, 42043, 11, 11044, 323, 6898, 594, 71435, 323, 56729, 11, 279, 26277, 9257, 315, 1039, 328, 8897, 389, 279, 3089, 4374, 20070, 11, 794, 13, 41508, 594, 56729, 11, 93348, 454, 859, 30296, 320, 1782, 2747, 315, 58030, 628, 258, 594, 9901, 8, 323, 803, 13, 4615, 2205, 7216, 8474, 686, 4446, 279, 3283, 594, 26291, 323, 6662, 815, 325, 3840, 311, 2272, 448, 7343, 323, 92966, 13, 1446, 686, 11877, 389, 279, 3283, 594, 10948, 323, 646, 1127, 11, 4669, 1378, 8606, 8522, 93630, 323, 25531, 325, 6133, 304, 419, 6233, 3283, 389, 419, 2613, 4351, 30184, 94340, 369, 17727, 304, 2635, 369, 220, 17, 2849, 13, 1096, 94340, 374, 43334, 12577, 369, 30451, 8284, 22172, 13, 5209, 3410, 25458, 1995, 518, 882, 315, 21857, 773, 429, 582, 1231, 4265, 279, 43334, 2630, 64, 1524, 2197, 4867, 311, 697, 18647, 624, 6667, 220, 17, 2849, 198, 54950, 315, 22950, 7989, 320, 52793, 1561, 74, 2253, 68, 328, 20172, 8, 323, 41563, 30296, 198, 26218, 279, 40692, 7324, 21682, 315, 279, 8522, 25076, 1561, 13, 362, 10655, 73717, 68, 3073, 6541, 686, 1896, 498, 311, 279, 6290, 315, 22950, 7989, 320, 52793, 1561, 74, 2253, 68, 328, 20172, 701, 1380, 498, 686, 1477, 264, 65841, 315, 8522, 3619, 60552, 12, 41563, 30296, 448, 279, 46664, 10420, 4766, 11, 892, 374, 6509, 311, 387, 825, 315, 279, 1879, 594, 39064, 13, 34713, 525, 7199, 311, 220, 24, 1251, 389, 419, 2613, 4351, 7216, 11, 22573, 498, 3278, 5258, 34549, 6529, 504, 697, 8474, 13, 1752, 803, 8522, 29236, 3840, 11, 13910, 697, 7216, 311, 2924, 42756, 35147, 4886, 30296, 624, 6667, 220, 20, 4115, 198, 623, 13, 53948, 220, 17, 54912, 44719, 14644, 25, 6252, 1763, 424, 11, 41563, 30296, 198, 1925, 419, 220, 17, 11228, 12752, 7216, 315, 794, 13, 53948, 11, 97544, 838, 504, 697, 30451, 8284, 323, 13186, 279, 3283, 748, 1429, 15978, 41166, 323, 12503, 32885, 13, 19008, 279, 1879, 2220, 22517, 6252, 1763, 424, 16328, 11, 13186, 8522, 29913, 10854, 2434, 11, 25191, 311, 279, 24209, 86355, 3795, 3936, 287, 11044, 75858, 77224, 5540, 11, 323, 728, 369, 264, 38921, 30451, 1365, 432, 748, 902, 11423, 429, 794, 13, 53948, 374, 1083, 3881, 438, 279, 1036, 58929, 558, 315, 279, 4787, 1987, 59445, 686, 5978, 697, 7216, 315, 279, 3283, 5780, 38411, 2337, 419, 220, 16, 22, 21231, 94340, 13, 1446, 646, 1083, 1841, 389, 369, 458, 5107, 11, 220, 18, 21231, 8894, 11, 2337, 892, 498, 646, 3139, 794, 13, 53948, 594, 5257, 92644, 38491, 11, 504, 279, 33496, 311, 97869, 23675, 624, 623, 13, 53948, 6252, 1763, 424, 16328, 25784, 10603, 91536, 28397, 323, 14644, 198, 43658, 279, 34751, 323, 5128, 518, 794, 53948, 748, 6252, 1763, 424, 16328, 448, 419, 220, 18, 21231, 7216, 4766, 825, 315, 279, 1879, 748, 12196, 1947, 42554, 13, 3085, 458, 6203, 8474, 11, 4669, 10706, 10603, 8447, 4343, 11, 323, 1948, 3217, 323, 6122, 11, 8722, 4124, 25293, 773, 498, 646, 1191, 697, 7216, 1573, 279, 34751, 13, 22656, 34549, 6529, 504, 697, 8474, 389, 419, 2613, 4351, 7216, 11, 7199, 311, 5779, 1251, 624, 40151, 14644, 315, 794, 53948, 198, 785, 10304, 14644, 315, 794, 53948, 6081, 15255, 279, 6638, 311, 1896, 304, 438, 1753, 438, 3204, 315, 279, 3283, 304, 264, 220, 18, 21231, 13929, 65054, 25, 264, 4583, 7216, 315, 794, 53948, 594, 1429, 11245, 6594, 504, 279, 6838, 315, 264, 1803, 476, 13420, 5242, 323, 448, 264, 40966, 323, 10321, 8474, 13, 576, 7216, 374, 458, 9073, 16800, 311, 794, 53948, 1177, 279, 2086, 7772, 323, 1429, 6233, 3283, 304, 8359, 13, 6065, 4558, 220, 18, 15, 15, 1635, 315, 1181, 3840, 794, 53948, 40065, 678, 279, 6662, 12559, 315, 279, 8522, 29913, 7154, 323, 6116, 825, 315, 279, 7772, 18652, 315, 7674, 323, 8038, 13, 34713, 525, 7199, 311, 4743, 1251, 389, 419, 2613, 4351, 7216, 11, 22573, 498, 3278, 5258, 34549, 6529, 504, 697, 8474, 624, 623, 53948, 44719, 38895, 34280, 25, 43334, 62890, 220, 17, 11228, 14644, 198, 1986, 2613, 4351, 320, 2810, 220, 16, 21, 1251, 8, 26655, 12577, 794, 53948, 46377, 38895, 34280, 686, 2968, 498, 458, 6638, 311, 1490, 678, 21314, 315, 794, 13, 53948, 624, 623, 13, 53948, 43334, 62890, 220, 17, 54912, 2009, 52200, 5737, 14644, 198, 38704, 419, 15817, 220, 17, 11228, 678, 27561, 13826, 1874, 7216, 518, 279, 15457, 3349, 3204, 11, 3602, 448, 264, 6838, 315, 11560, 264, 2613, 13420, 1455, 355, 448, 3807, 1008, 22172, 13, 1096, 374, 264, 43334, 62890, 2473, 892, 3363, 498, 653, 537, 1184, 311, 6851, 264, 8522, 43334, 2567, 13, 5737, 1379, 374, 7199, 311, 220, 16, 21, 1251, 624, 623, 13, 53948, 220, 17, 54912, 10304, 44719, 38895, 34280, 14644, 198, 1986, 2613, 1874, 320, 39187, 220, 16, 21, 1251, 701, 678, 3419, 8336, 1378, 1899, 6662, 7216, 6081, 279, 1429, 76988, 21314, 315, 794, 13, 53948, 369, 1846, 879, 6426, 311, 1281, 279, 1429, 315, 862, 220, 17, 2849, 304, 419, 6233, 323, 13245, 3283, 323, 374, 279, 4727, 1616, 311, 13186, 794, 13, 53948, 0, 44052, 279, 1850, 315, 40692, 794, 13, 53948, 304, 264, 2613, 1874, 7216, 11, 10693, 369, 264, 803, 34549, 323, 31387, 3139, 448, 14848, 2473, 2670, 93630, 11, 264, 38921, 15328, 11877, 13, 576, 220, 17, 1899, 6662, 7216, 14521, 678, 315, 279, 1429, 76988, 1969, 12, 4060, 41166, 315, 794, 13, 53948, 448, 1039, 12235, 11, 11657, 11, 1424, 2268, 18504, 27193, 879, 4446, 311, 2272, 279, 26291, 41166, 323, 3840, 315, 279, 3283, 13, 1096, 15817, 7216, 374, 10507, 369, 14709, 879, 6426, 311, 1281, 279, 1429, 315, 862, 220, 17, 1899, 4717, 4894, 623, 13, 53948, 323, 19243, 10080, 16328, 220, 17, 54912, 14644, 198, 1986, 1378, 11228, 1874, 94340, 374, 264, 2244, 1616, 311, 1490, 279, 1429, 1632, 21309, 41166, 315, 794, 13, 53948, 3156, 448, 279, 19243, 10080, 23971, 11, 892, 702, 3635, 825, 315, 279, 501, 42158, 304, 279, 3283, 13, 34702, 504, 419, 23971, 11, 498, 3278, 3947, 1378, 29236, 84771, 11, 2670, 279, 41563, 30296, 304, 22950, 7989, 11, 323, 11044, 75858, 448, 77224, 6118, 13, 7281, 11, 1490, 279, 4766, 315, 279, 9257, 389, 279, 3089, 4374, 20070, 323, 794, 13, 41508, 594, 56729, 13, 3776, 315, 279, 3598, 38491, 498, 2291, 264, 3947, 311, 374, 7838, 279, 6252, 1763, 424, 11, 279, 7772, 23971, 304, 279, 3146, 13, 11954, 419, 7216, 498, 686, 3960, 7040, 13064, 911, 279, 3042, 323, 3267, 3840, 315, 279, 3283, 11, 6723, 279, 7343, 911, 279, 12751, 859, 2997, 323, 8722, 264, 19117, 8660, 315, 8522, 7674, 11, 30906, 323, 17778, 2272, 13, 5633, 27193, 686, 1281, 279, 3840, 315, 279, 3283, 2525, 13675, 323, 1039, 6584, 2083, 686, 653, 4297, 311, 1281, 697, 882, 304, 419, 6233, 3283, 1602, 3281, 13}));
  }
  {
    LOG(INFO) << "========= Zero ===========";
    //std::string text = "Consumer Discretionary : Specialty Retail | Small Cap Value\nGenesco Inc. is a retailer and wholesaler of footwear, apparel and accessories. The Company operates in four segments: Journeys Group, Schuh Group, Johnston & Murphy Group and Licensed Brands. It relies on independent third-party manufacturers for production of its footwear products sold at wholesale. It sources footwear and accessory products from foreign manufacturers located in Bangladesh, Brazil, Cambodia, Canada, China, Dominican Republic, El Salvador, France, Germany, Hong Kong, India, Indonesia, Italy, Mexico, the Netherlands, Portugal, Peru, Romania, Taiwan and Vietnam. As of January 28, 2017, it operated 2,794 retail footwear, headwear and sports apparel and accessory stores and leased departments located primarily throughout the United States and in Puerto Rico, including 147 headwear and sports apparel and accessory stores and 87 footwear stores in Canada and 128 footwear stores in the United Kingdom, the Republic of Ireland and Germany.\u0000\nToday's volume of 28,252 shares is on pace to be much lighter than GCO's 10-day average volume of 261,247 shares.";
    const char buf[] = "Consumer Discretionary : Specialty Retail | Small Cap Value\nGenesco Inc. is a retailer and wholesaler of footwear, apparel and accessories. The Company operates in four segments: Journeys Group, Schuh Group, Johnston & Murphy Group and Licensed Brands. It relies on independent third-party manufacturers for production of its footwear products sold at wholesale. It sources footwear and accessory products from foreign manufacturers located in Bangladesh, Brazil, Cambodia, Canada, China, Dominican Republic, El Salvador, France, Germany, Hong Kong, India, Indonesia, Italy, Mexico, the Netherlands, Portugal, Peru, Romania, Taiwan and Vietnam. As of January 28, 2017, it operated 2,794 retail footwear, headwear and sports apparel and accessory stores and leased departments located primarily throughout the United States and in Puerto Rico, including 147 headwear and sports apparel and accessory stores and 87 footwear stores in Canada and 128 footwear stores in the United Kingdom, the Republic of Ireland and Germany.\u0000\nToday's volume of 28,252 shares is on pace to be much lighter than GCO's 10-day average volume of 261,247 shares.";
    std::string text(buf, sizeof(buf) - 1);
    LOG(INFO) << "Len[" << text.size() << "]";
    auto text_normal = mycommon::utf8_normalize(text);
    LOG(INFO) << "Nomal Len[" << text_normal.size() << "]";
    auto ids = token.encode(text_normal);
    LOG(INFO) << mycommon::str_join(ids);
    ASSERT_EQ(true, ids == std::vector<size_t>({29968, 4093, 88690, 658, 549, 81514, 34039, 760, 14994, 8012, 5162, 198, 9967, 93834, 4848, 13, 374, 264, 36791, 323, 25211, 13111, 315, 67072, 11, 54325, 323, 22293, 13, 576, 8188, 26057, 304, 3040, 20632, 25, 619, 413, 35271, 5737, 11, 5016, 12540, 5737, 11, 60482, 609, 29953, 5737, 323, 10103, 54232, 13, 1084, 33644, 389, 9489, 4843, 24031, 16621, 369, 5670, 315, 1181, 67072, 3871, 6088, 518, 34457, 13, 1084, 8173, 67072, 323, 41981, 3871, 504, 7214, 16621, 7407, 304, 38501, 11, 15948, 11, 61038, 11, 6864, 11, 5616, 11, 66013, 5429, 11, 3984, 48359, 11, 9625, 11, 9856, 11, 19180, 18211, 11, 6747, 11, 23968, 11, 15344, 11, 12270, 11, 279, 25662, 11, 33311, 11, 47747, 11, 46049, 11, 28289, 323, 22500, 13, 1634, 315, 6058, 220, 17, 23, 11, 220, 17, 15, 16, 22, 11, 432, 23151, 220, 17, 11, 22, 24, 19, 10806, 67072, 11, 1968, 22744, 323, 9833, 54325, 323, 41981, 10533, 323, 81180, 25215, 7407, 15503, 6814, 279, 3639, 4180, 323, 304, 30219, 33148, 11, 2670, 220, 16, 19, 22, 1968, 22744, 323, 9833, 54325, 323, 41981, 10533, 323, 220, 23, 22, 67072, 10533, 304, 6864, 323, 220, 16, 17, 23, 67072, 10533, 304, 279, 3639, 15072, 11, 279, 5429, 315, 14648, 323, 9856, 13, 188, 198, 15364, 594, 8123, 315, 220, 17, 23, 11, 17, 20, 17, 13248, 374, 389, 17857, 311, 387, 1753, 29573, 1091, 479, 8281, 594, 220, 16, 15, 11228, 5461, 8123, 315, 220, 17, 21, 16, 11, 17, 19, 22, 13248, 13}));
  }
#endif
  {
    LOG(INFO) << "=== Wrong ===";
    #if 0
    std::string filepath = "/tmp/a.txt";
    std::ifstream file(filepath);
    ASSERT_EQ(true, file.is_open());
    std::string line;
    while (std::getline(file, line)) {
      mycommon::str_trim(line, "\r\n\t", false, true);
      std::string text;
      std::vector<size_t> ids;
      {
        rapidjson::Document doc; 
        rapidjson::ParseResult ok = doc.Parse(line.c_str());
        if (ok && doc.IsObject()) {
          // Support \u0000
          text = GET_JSON_STRING_BINARY(doc, "text", "");
          if (doc.HasMember("ids") && doc["ids"].IsArray()) {
            for (int j = 0; j < doc["ids"].Size(); j++) {
              rapidjson::Value& json_id = doc["ids"][j];
              if (json_id.IsInt()) {
                ids.push_back(json_id.GetInt());
              }
              else {
                ASSERT_EQ(false, true);
              }
            }
          }
        }
      }
      if (text.empty()) {
        ASSERT_EQ(false, text.empty());
      }
      mycommon::file_write("/tmp/b.txt", text);
      std::vector<size_t> my_ids = token.encode(mycommon::utf8_normalize(text));
      //LOG(INFO) << mycommon::str_join(my_ids);
      //LOG(INFO) << mycommon::str_join(ids);
      ASSERT_EQ(my_ids, ids);
    }
    #endif
    std::string text = "generating pronouns as the first word followed by \\texttt{<|endoftext|>}. Nonetheless, the trend of female pronouns being the least favorable is";
    std::vector<size_t> my_ids = token.encode(mycommon::utf8_normalize(text));
    LOG(INFO) << mycommon::str_join(my_ids);
    ASSERT_EQ(my_ids, std::vector<size_t>({7490,1095,18613,58986,438,279,1156,3409,8110,553,1124,1318,5566,90,151643,7810,55633,11,279,9149,315,8778,18613,58986,1660,279,3245,36749,374}));
    ASSERT_EQ(token.encode(mycommon::utf8_normalize("{<|endoftext|>}")), std::vector<size_t>({90,151643,92}));
    ASSERT_EQ(token.encode(mycommon::utf8_normalize("{<|endoftext|>}word followed by \\texttt{<|endoftext|>}. Nonetheless,")), std::vector<size_t>({90,151643,92,1158,8110,553,1124,1318,5566,90,151643,7810,55633,11}));
  }
}

TEST(Token, QwenDecodeBase) {
  QwenToken token("/data/Qwen3-0.6B/");
  ASSERT_EQ(true, token.encode("Hello") == std::vector<size_t>({9707}));
  ASSERT_EQ(true, token.decode(std::vector<size_t>({9707})) == "Hello");
  ASSERT_EQ(true, token.encode("&amp;लेबल") == std::vector<size_t>({27066, 26, 91811, 54784, 105, 91811}));
  ASSERT_EQ(true, token.decode(std::vector<size_t>({27066, 26, 91811, 54784, 105, 91811})) == "&amp;लेबल");
  ASSERT_EQ(true, token.encode(mycommon::utf8_normalize("{<|endoftext|>}word followed by \\texttt{<|endoftext|>}. Nonetheless,")) == std::vector<size_t>({90,151643,92,1158,8110,553,1124,1318,5566,90,151643,7810,55633,11}));
  ASSERT_EQ(true, token.decode(std::vector<size_t>({90,151643,92,1158,8110,553,1124,1318,5566,90,151643,7810,55633,11})) == mycommon::utf8_normalize("{<|endoftext|>}word followed by \\texttt{<|endoftext|>}. Nonetheless,"));
}

TEST(Token, QwenEncodeBatch) {
  QwenToken token("/data/Qwen3-0.6B/");
  std::vector<std::tuple<int, std::string, std::vector<size_t>>> lines;
  {
    std::string filepath = "/data/token_batch_data.jsonl";
    //std::string filepath = "/tmp/3.txt";
    std::ifstream file(filepath);
    ASSERT_EQ(true, file.is_open());
    int line_index = -1;
    std::string line;
    while (std::getline(file, line)) {
      line_index++;
      mycommon::str_trim(line, "\r\n\t", false, true);
      std::string text;
      std::vector<size_t> ids;
      {
        rapidjson::Document doc; 
        rapidjson::ParseResult ok = doc.Parse(line.c_str());
        if (ok && doc.IsObject()) {
          // Support \u0000
          text = GET_JSON_STRING_BINARY(doc, "text", "");
          if (doc.HasMember("ids") && doc["ids"].IsArray()) {
            for (int j = 0; j < doc["ids"].Size(); j++) {
              rapidjson::Value& json_id = doc["ids"][j];
              if (json_id.IsInt()) {
                ids.push_back(json_id.GetInt());
              }
              else {
                LOG(INFO) << "Id Invalid[" << line_index << "]";
                ASSERT_EQ(false, true);
              }
            }
          }
        }
      }
      if (text.empty()) {
        LOG(INFO) << "Text Empty[" << line_index << "]";
        ASSERT_EQ(false, text.empty());
      }
      lines.emplace_back(line_index, mycommon::utf8_normalize(text), ids);
      //if (lines.size() >= 30000) {
      //  break;
      //}
    }
    file.close();
    LOG(INFO) << "Total[" << lines.size() << "]";
  }
#if 0
  int line_index = -1;
  int fail_ct = 0;
  LOG(INFO) << "Line[" << line.size() << "] Text[" << text.size() << "]";
  //mycommon::file_write("/tmp/4.txt", text);
  std::vector<size_t> my_ids = token.encode(text);
  if (ids != my_ids) {
    LOG(INFO) << "Dont Equal[" << line_index << "]";
    //ASSERT_EQ(true, false);
    fail_ct++;
    //std::string tmp_str;
    //for (int i = 0; i < my_ids.size(); i++) {
    //  tmp_str += std::to_string(my_ids[i]);
    //  tmp_str += ",";
    //}
    //fprintf(stderr, "%s\n", tmp_str.c_str());
    //break;
  }
  LOG(INFO) << "Total[" << (line_index + 1) << "] Fail[" << fail_ct << "]";
#endif
  std::atomic<int> running_ct = {0};
  std::atomic<int> fail_ct = {0};
	RunMulti<std::tuple<int, std::string, std::vector<size_t>>>(lines, 4, [&](const std::tuple<int, std::string, std::vector<size_t>>& info) {
    std::vector<size_t> my_ids = token.encode(std::get<1>(info));
    if (std::get<2>(info) != my_ids) {
      fail_ct++;
      LOG(INFO) << "Dont Equal[" << std::get<0>(info) << "] Ori[" << mycommon::str_join(std::get<2>(info)) << "] My[" << mycommon::str_join(my_ids) << "]";
    }
    running_ct++;
    if (0 == (running_ct % 1000)) {
      LOG(INFO) << "Running[" << running_ct << "]";
    }
  });
  LOG(INFO) << "Total[" << lines.size() << "] Fail[" << fail_ct << "]";
}

TEST(Token, QwenDecodeBatch) {
  QwenToken token("/data/Qwen3-0.6B/");
  std::vector<std::tuple<int, std::string, std::vector<size_t>>> lines;
  {
    std::string filepath = "/data/token_batch_data.jsonl";
    std::ifstream file(filepath);
    ASSERT_EQ(true, file.is_open());
    int line_index = -1;
    std::string line;
    while (std::getline(file, line)) {
      line_index++;
      mycommon::str_trim(line, "\r\n\t", false, true);
      std::string text;
      std::vector<size_t> ids;
      {
        rapidjson::Document doc; 
        rapidjson::ParseResult ok = doc.Parse(line.c_str());
        if (ok && doc.IsObject()) {
          // Support \u0000
          text = GET_JSON_STRING_BINARY(doc, "text", "");
          if (doc.HasMember("ids") && doc["ids"].IsArray()) {
            for (int j = 0; j < doc["ids"].Size(); j++) {
              rapidjson::Value& json_id = doc["ids"][j];
              if (json_id.IsInt()) {
                ids.push_back(json_id.GetInt());
              }
              else {
                LOG(INFO) << "Id Invalid[" << line_index << "]";
                ASSERT_EQ(false, true);
              }
            }
          }
        }
      }
      if (text.empty()) {
        LOG(INFO) << "Text Empty[" << line_index << "]";
        ASSERT_EQ(false, text.empty());
      }
      lines.emplace_back(line_index, mycommon::utf8_normalize(text), ids);
    }
    file.close();
    LOG(INFO) << "Total[" << lines.size() << "]";
  }
  std::atomic<int> running_ct = {0};
  std::atomic<int> fail_ct = {0};
	RunMulti<std::tuple<int, std::string, std::vector<size_t>>>(lines, 16, [&](const std::tuple<int, std::string, std::vector<size_t>>& info) {
    std::string my_text = token.decode(std::get<2>(info));
    if (std::get<1>(info) != my_text) {
      fail_ct++;
      LOG(INFO) << "Dont Equal[" << std::get<0>(info) << "] Ori[" << std::get<1>(info) << "] My[" << my_text << "]";
    }
    running_ct++;
    if (0 == (running_ct % 1000)) {
      LOG(INFO) << "Running[" << running_ct << "]";
    }
  });
  LOG(INFO) << "Total[" << lines.size() << "] Fail[" << fail_ct << "]";
}

TEST(Token, Infer) {
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */

#pragma once
#include <filesystem>
#include <stdexcept>
namespace ms {
inline std::filesystem::path photo_file(const std::filesystem::path& dir,const std::string& stem){
 std::filesystem::path found;
 for(auto extension:{".jpg",".png",".jpeg"}){auto p=dir/(stem+extension);if(std::filesystem::is_regular_file(p)){if(!found.empty())throw std::runtime_error("ambiguous photo stem: "+stem);found=p;}}
 if(found.empty())throw std::runtime_error("photo missing: "+stem);
 return found;
}
}

#pragma once
#include <array>
#include <stdexcept>
namespace native_calibration {
// Metashape 2.2.2, SHA256 90557da3...; both masked-data traces agree.
// This describes parameter scheduling, not the whole proprietary SfM pipeline.
enum class Stage { Candidate, Incremental, FullPrior, FinalFull };
struct Policy {unsigned accepted,tentative,iterations,rounds;bool adaptive;const char* name;};
constexpr Policy policy(Stage stage){
 switch(stage){
 case Stage::Candidate:return {193,193,20,5,false,"candidate"};
 case Stage::Incremental:return {65,193,10,5,true,"incremental"};
 case Stage::FullPrior:return {65,3527,20,5,true,"full_prior"};
 case Stage::FinalFull:return {3527,3527,20,5,false,"final_full"};
 }
 throw std::invalid_argument("unknown calibration stage");
}
constexpr unsigned fitted_mask(Policy p){return p.adaptive?p.tentative:p.accepted;}
constexpr std::array<unsigned,8> supported_bits{0,1,2,6,7,8,10,11};
}

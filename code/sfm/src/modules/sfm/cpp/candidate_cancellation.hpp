#pragma once
#include <atomic>
#include <tuple>
inline auto initial_pair_pool_key(unsigned score,unsigned common,unsigned first,unsigned second){return std::tuple{score,common,first,second};}
template<class Score> auto initial_candidate_key(const Score&s,unsigned first,unsigned second){return std::tuple{s.aligned,s.points>=100,s.accuracy>=.0025,-s.error,s.accuracy,s.points,first,second};}
// 142EAF655 requests cancellation; 142EA8330 observes it and throws;
// the worker's exception wrapper then closes publication. These are two states.
struct CandidateCancelled {};
struct CandidateCancellation {
 std::atomic<bool> requested{},aborted{};
 bool can_publish()const{return !aborted.load();}
 void request(){requested.store(true);}
 void checkpoint(){if(requested.load())throw CandidateCancelled{};}
 void close_after_abort(){aborted.store(true);}
};
inline thread_local CandidateCancellation* current_candidate_cancellation=nullptr;
inline void candidate_checkpoint(){if(current_candidate_cancellation)current_candidate_cancellation->checkpoint();}

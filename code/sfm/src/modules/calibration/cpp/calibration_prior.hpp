#pragma once
#include "bundle_model.hpp"
namespace bundle {
using IntrinsicPrior=std::array<std::byte,520>;
struct PriorSetup {unsigned mask{};std::vector<IntrinsicPrior> priors;};
// 142EEA570 / 142EDA7A0. Inputs precede scene construction; no solved prior is read.
inline PriorSetup construct_prior(const std::byte* master,bool adaptive,unsigned sensor=0){
 unsigned blocked=get<unsigned>(master,856)|get<unsigned>(master,860);
 unsigned old=get<unsigned>(master,852)&~blocked,next=get<unsigned>(master,864)&~blocked;
 PriorSetup result{old,{}};
 if(!adaptive||master[848]==std::byte{}||old==next)return result;
 const auto* c=master+272;double f=get<double>(c,24),b=get<double>(c,48);
 double y=(get<double>(master,552)-get<double>(master,536))*0.5/f;
 double x=(get<double>(master,544)-get<double>(master,528))*0.5/(f+b);
 double r=std::sqrt(y*y+x*x),r2=r*r;
 if(!(r>0)||!std::isfinite(r))throw std::runtime_error("invalid prior image extent");
 const std::array<unsigned,13> bits{0,1,2,3,4,6,7,8,9,10,11,12,13};
 const double tiny_scale=std::bit_cast<double>(std::uint64_t{0x3DDB7CDFD9D7BDBB});
 const std::array<double,13> scale{1/r,1,1,1/r,1/r,1/(r2*f*r),1/(r2*r2*f*r),1/(r2*r2*r2*f*r),1/(r2*r2*(r2*r2)*f*r),1/(r2*f),1/(r2*f),tiny_scale,tiny_scale};
 IntrinsicPrior prior{};set(prior.data(),0,sensor);std::memcpy(prior.data()+8,c,256);
 for(unsigned i=0;i<bits.size();++i){auto bit=bits[i],offset=24+8*bit+(bit>=6?8:0);double width=scale[i]*((old&(1u<<bit))?1000000.0:0.01);set(prior.data(),264+offset,1.0/width);}
 result.mask=next;result.priors.push_back(prior);return result;
}
struct PriorEvaluation {std::vector<double> residual,diagonal;double cost{};};
// 142EE7270: a sensor index, a 256-byte target calibration and matching weights.
// Only fitted entries contribute, even when other prior entries are nonzero.
inline PriorEvaluation evaluate_prior(const std::byte* calibration,const IntrinsicPrior& prior,std::span<const unsigned> parameters){
 if(get<unsigned>(prior.data(),0)!=0)throw std::runtime_error("prior requires another sensor");
 PriorEvaluation result;double sum=0;
 for(auto bit:parameters){if(bit>=18||bit==5)throw std::runtime_error("unsupported prior parameter");auto offset=24+8*bit+(bit>=6?8:0);double target=get<double>(prior.data(),offset+8),weight=get<double>(prior.data(),offset+264),current=get<double>(calibration,offset);if(!std::isfinite(target)||!std::isfinite(weight))throw std::runtime_error("nonfinite calibration prior");double value=(current-target)*weight;sum+=value*value;result.residual.push_back((current-target)*(-weight));result.diagonal.push_back(weight);}
 result.cost=sum*0.5;return result;
}
}


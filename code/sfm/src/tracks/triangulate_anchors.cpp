#define GRAPH6_NO_MAIN
#include "refine_graph.cpp"

namespace anchor {
double objective(const sfm::Reconstruction&r,std::span<const sfm::Obs>obs,V x){double value=0;for(auto o:obs){double e=sfm::geometry_error(r,o,x);if(!std::isfinite(e))return INFINITY;value+=2*std::log1p(e*e/4);}return value;}
V optimize(const sfm::Reconstruction&r,std::span<const sfm::Obs>obs,V x){
 double value=objective(r,obs,x),lambda=1e-4;
 for(unsigned it=0;it<40&&std::isfinite(value);++it){M h{};V g{};
  for(auto o:obs){auto e=sfm::point_residual(r,o,x);double w=1/(1+(e[0]*e[0]+e[1]*e[1])/4);std::array<std::array<double,2>,3>j{};
   for(unsigned k=0;k<3;++k){double step=1e-6*std::max(1.,std::abs(x[k]));auto a=x,b=x;a[k]+=step;b[k]-=step;auto ea=sfm::point_residual(r,o,a),eb=sfm::point_residual(r,o,b);for(unsigned l=0;l<2;++l)j[k][l]=(ea[l]-eb[l])/(2*step);}
   for(unsigned k=0;k<3;++k){g[k]+=w*(j[k][0]*e[0]+j[k][1]*e[1]);for(unsigned l=0;l<3;++l)h[k*3+l]+=w*(j[k][0]*j[l][0]+j[k][1]*j[l][1]);}
  }for(unsigned k=0;k<3;++k)h[k*3+k]+=lambda*std::max(h[k*3+k],1e-12);
  auto next=sub(x,mv(inverse3(h),g));double proposed=objective(r,obs,next);
  if(proposed<value){double gain=value-proposed;x=next;value=proposed;lambda=std::max(1e-12,lambda*.2);if(gain<1e-10*std::max(1.,value))break;}else{lambda*=10;if(lambda>1e10)break;}
 }return x;
}
bool refine(const sfm::Reconstruction&r,sfm::Track&t){
 if(t.obs.size()<2)return false;V best{};double cost=INFINITY;std::vector<std::pair<double,V>>candidates;
 auto proposal=[&](V x){double c=objective(r,t.obs,x);if(std::isfinite(c))candidates.push_back({c,x});};
 proposal(sfm::intersect_track(r,t.obs));auto old=t;sfm::RobustTrackStats stats;if(sfm::recover_track(r,old,stats))proposal(old.xyz);
 // All pairs for short tracks, evenly spaced deterministic pairs for long ones.
 std::vector<std::pair<unsigned,unsigned>>pairs;for(unsigned i=0;i<t.obs.size();++i)for(unsigned j=i+1;j<t.obs.size();++j)pairs.push_back({i,j});
 unsigned count=std::min(64u,unsigned(pairs.size()));for(unsigned k=0;k<count;++k){auto [a,b]=pairs[count<=1?0:std::uint64_t(k)*(pairs.size()-1)/(count-1)];std::array<sfm::Obs,2>obs{t.obs[a],t.obs[b]};auto x=sfm::intersect_track(r,obs);if(sfm::sufficient_track_angle(r,obs,x))proposal(x);}
 std::sort(candidates.begin(),candidates.end(),[](auto&a,auto&b){return a.first<b.first;});
 for(unsigned i=0;i<std::min(4u,unsigned(candidates.size()));++i){auto x=optimize(r,t.obs,candidates[i].second);double c=objective(r,t.obs,x);if(c<cost){cost=c;best=x;}}
 if(!std::isfinite(cost)||!sfm::sufficient_track_angle(r,t.obs,best))return false;t.xyz=best;t.valid=true;return true;
}
struct Result {unsigned track{},valid{};V xyz{};};static_assert(sizeof(Result)==32);
}
int wmain(int argc,wchar_t**argv){try{
 if(argc!=5||fs::exists(argv[3]))throw std::runtime_error("usage: triangulate_anchors INPUT_SFM ANCHORS_BIN NEW_POINTS native|cauchy");
 auto r=calibrated::load_model(argv[1]);r.tracks.clear();for(auto o:read_array<sfm::Obs>(argv[2])){if(!r.map.contains(o.camera))throw std::runtime_error("unknown camera");r.tracks[o.track].obs.push_back(o);}
 std::wstring mode=argv[4];if(mode!=L"native"&&mode!=L"cauchy")throw std::runtime_error("unknown mode");std::vector<anchor::Result>result;
 for(auto&[id,t]:r.tracks){sfm::RobustTrackStats stats;bool ok=mode==L"native"?sfm::recover_track(r,t,stats):anchor::refine(r,t);result.push_back({id,unsigned(ok),t.xyz});}
 write_array(argv[3],result);return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}

#define GRAPH6_NO_MAIN
#include "refine_graph.cpp"

// Ordered-capture hypothesis: a track may drift onto a nearby repeated texture.
// Split only if two contiguous segments predict training-only held-out pixels
// better than a single point. No benchmark query is accessible here.
namespace segments {
using Obs=sfm::Obs;
struct Result{bool accepted{};unsigned cut{};V a{},b{};double before{},after{},cv_before{},cv_after{};};
bool fit(const sfm::Reconstruction&r,std::span<const Obs>o,V&x){
 if(o.size()<3)return false;x=sfm::refine_track_point(r,o,sfm::intersect_track(r,o));
 return sfm::sufficient_track_angle(r,o,x)&&std::all_of(o.begin(),o.end(),[&](auto p){return std::isfinite(sfm::geometry_error(r,p,x));});
}
bool cv(const sfm::Reconstruction&r,const std::vector<Obs>&o,unsigned cut,double&one,double&two){
 one=two=0;
 for(unsigned parity=0;parity<2;++parity){std::vector<Obs>all,a,b,qa,qb;
  for(unsigned i=0;i<o.size();++i){if(i%2==parity){all.push_back(o[i]);(i<cut?a:b).push_back(o[i]);}else(i<cut?qa:qb).push_back(o[i]);}
  V x,y,z;if(!fit(r,all,x)||!fit(r,a,y)||!fit(r,b,z))return false;
  for(auto p:qa){double e=sfm::geometry_error(r,p,x),f=sfm::geometry_error(r,p,y);one+=e*e;two+=f*f;}
  for(auto p:qb){double e=sfm::geometry_error(r,p,x),f=sfm::geometry_error(r,p,z);one+=e*e;two+=f*f;}
 }return std::isfinite(one)&&std::isfinite(two);
}
Result choose(const sfm::Reconstruction&r,const std::vector<Obs>&o){
 Result best;if(o.size()<12)return best;V x;if(!fit(r,o,x))return best;double base=sfm::point_cost(r,o,x);
 // Pixel sigma fixed at 1.5, 3 extra XYZ parameters and candidate-search penalty.
 double penalty=2.25*(3*std::log(2.*o.size())+2*std::log(3.));
 for(unsigned cut:std::set<unsigned>{unsigned(o.size()/3),unsigned(o.size()/2),unsigned(2*o.size()/3)}){
  if(cut<6||o.size()-cut<6)continue;auto a=std::span(o).first(cut),b=std::span(o).subspan(cut);V xa,xb;
  if(!fit(r,a,xa)||!fit(r,b,xb))continue;double after=sfm::point_cost(r,a,xa)+sfm::point_cost(r,b,xb);
  if(!(base-after>penalty&&after<.5*base))continue;double c1,c2;if(!cv(r,o,cut,c1,c2)||c2>=.75*c1)continue;
  if(!best.accepted||after<best.after)best={true,cut,xa,xb,base,after,c1,c2};
 }return best;
}
void split(sfm::Reconstruction&r,std::ostream&log){
 unsigned next=r.tracks.rbegin()->first+1,accepted=0,eligible=0;std::vector<std::pair<unsigned,sfm::Track>>children;
 for(auto&[id,t]:r.tracks)if(t.valid&&t.obs.size()>=12){++eligible;auto o=t.obs;std::sort(o.begin(),o.end(),[](auto a,auto b){return a.camera<b.camera;});auto q=choose(r,o);if(!q.accepted)continue;
  unsigned child=next++;sfm::Track other;other.valid=true;other.xyz=q.b;other.obs.assign(o.begin()+q.cut,o.end());for(auto&v:other.obs)v.track=child;
  t.obs.assign(o.begin(),o.begin()+q.cut);t.xyz=q.a;children.emplace_back(child,std::move(other));++accepted;
  log<<"{\"split\":true,\"parent\":"<<id<<",\"child\":"<<child<<",\"cut\":"<<q.cut<<",\"views\":"<<o.size()<<",\"sse_before\":"<<q.before<<",\"sse_after\":"<<q.after<<",\"cv_before\":"<<q.cv_before<<",\"cv_after\":"<<q.cv_after<<"}\n";
 }
 for(auto&[id,t]:children)r.tracks.emplace(id,std::move(t));
 log<<"{\"split_summary\":true,\"eligible\":"<<eligible<<",\"accepted\":"<<accepted<<",\"deleted_observations\":0}\n";log.flush();
}
void test(){
 sfm::Reconstruction r;initial_calibration(r.s,3072,2048);std::vector<Obs>one,two;
 for(unsigned i=0;i<20;++i){Pose p;p.rotation=identity;set(p.position.data(),96,V{-.5+.05*i,.06*std::sin(i*.4),0});sfm::add(r,i,p);
  for(unsigned mode=0;mode<2;++mode){V x{.1,.05,3};if(mode&&i>=10)x[1]+=.02;Obs o{};o.camera=i;o.track=0;o.size=1;auto e=evaluate(r.s.calibration.data(),p,x,o,false).residual;o.u=float(-e[0]+.05*std::sin(i));o.v=float(-e[1]+.05*std::cos(i));(mode?two:one).push_back(o);}}
 if(choose(r,one).accepted)throw std::runtime_error("clean synthetic track oversplit");
 if(!choose(r,two).accepted)throw std::runtime_error("synthetic drift not split");
 sfm::Track t;t.valid=true;t.obs=two;t.xyz={.1,.05,3};r.tracks.emplace(0,t);std::ostringstream log;split(r,log);
 unsigned n=0;std::set<unsigned>cams;for(auto&[id,v]:r.tracks)for(auto o:v.obs){++n;if(o.track!=id||!cams.insert(o.camera).second)throw std::runtime_error("ownership changed");}
 if(n!=20||r.tracks.size()!=2)throw std::runtime_error("observation count changed");std::cout<<"PASS clean-track rejection, synthetic drift splitting and unique observation preservation\n";
}
}
int wmain(int argc,wchar_t**argv){try{
 if(argc==2&&std::wstring(argv[1])==L"--self-test"){segments::test();return 0;}
 if(argc!=3||fs::exists(argv[2]))throw std::runtime_error("usage: refine_track_segments INPUT_SFM NEW_OUTPUT");
 auto r=calibrated::load_model(argv[1]);auto cal=r.s.calibration;fs::path out=argv[2];fs::create_directories(out);std::ofstream log(out/"optimization.jsonl");log<<std::setprecision(17);
 segments::split(r,log);calibrated::Policy p;p.target=cal;p.observation_gate=1e6;
 for(unsigned round=0;round<3;++round){log<<"{\"round\":"<<round<<",\"ba_call\":true}\n";calibrated::refine(r,p,log);}
 if(r.s.calibration!=cal)throw std::runtime_error("intrinsics changed");graph6::save_model(r,argv[1],out);return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}

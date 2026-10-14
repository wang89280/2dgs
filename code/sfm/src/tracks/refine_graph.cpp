#define CALIBRATED_NO_MAIN
#include "refine_calibration.cpp"
#include <bit>

namespace graph6 {
constexpr unsigned none=UINT_MAX;
struct Node { unsigned camera,feature; float u,v; };
static_assert(sizeof(Node)==16);
using Edge=std::array<unsigned,2>;
using Key=std::tuple<unsigned,unsigned,unsigned>;
Key key(unsigned camera,float u,float v){return {camera,std::bit_cast<unsigned>(u),std::bit_cast<unsigned>(v)};}

struct Graph {
 std::vector<Node> nodes;
 std::vector<Edge> edges;
 std::vector<std::vector<unsigned>> adjacent;
 std::map<Key,unsigned> lookup;
 std::vector<unsigned> owner;
 std::map<unsigned,std::vector<unsigned>> members;
 void initialize(){
  adjacent.assign(nodes.size(),{});lookup.clear();
  std::set<std::pair<unsigned,unsigned>> features;
  for(unsigned i=0;i<nodes.size();++i){auto n=nodes[i];if(!lookup.emplace(key(n.camera,n.u,n.v),i).second||!features.emplace(n.camera,n.feature).second)throw std::runtime_error("ambiguous or duplicate feature node");}
  std::set<Edge> unique;
  for(auto edge:edges){auto [a,b]=edge;if(a>=nodes.size()||b>=nodes.size()||a==b||nodes[a].camera==nodes[b].camera)throw std::runtime_error("invalid verified edge");if(a>b)std::swap(a,b);if(!unique.insert({a,b}).second)throw std::runtime_error("duplicate verified edge");adjacent[a].push_back(b);adjacent[b].push_back(a);}
 }
 void refresh(const sfm::Reconstruction&r){
  owner.assign(nodes.size(),none);members.clear();
  for(const auto&[id,t]:r.tracks)if(t.valid){std::set<unsigned> cameras;for(auto o:t.obs){auto found=lookup.find(key(o.camera,o.u,o.v));if(found==lookup.end())throw std::runtime_error("observation absent from allowed training graph");auto n=found->second;if(owner[n]!=none||!cameras.insert(o.camera).second)throw std::runtime_error("feature ownership or camera conflict");owner[n]=id;members[id].push_back(n);}}
 }
 sfm::Obs observation(unsigned node,unsigned track)const{auto n=nodes.at(node);sfm::Obs o{};o.camera=n.camera;o.track=track;o.auxiliary=UINT64_MAX;o.u=n.u;o.v=n.v;o.size=1;return o;}
 unsigned support(unsigned node,unsigned track)const {unsigned n=0;for(auto other:adjacent.at(node))n+=owner[other]==track;return n;}
};

// Every accepted graph change keeps original pixel coordinates and unique ownership.
// No coordinate averaging and no merging by proximity in 3D.
bool compatible_cameras(std::span<const sfm::Obs>a,std::span<const sfm::Obs>b){std::set<unsigned> used;for(auto o:a)used.insert(o.camera);for(auto o:b)if(!used.insert(o.camera).second)return false;return true;}
bool fit_all(const sfm::Reconstruction&r,std::vector<sfm::Obs>&obs,V seed,V&xyz,double gate=4,double rms_limit=2.25){
 xyz=sfm::refine_track_point(r,obs,seed);double sum=0;
 for(auto o:obs){double e=sfm::geometry_error(r,o,xyz);if(e>gate)return false;sum+=e*e;}
 return !obs.empty()&&sum<=rms_limit*rms_limit*obs.size()&&sfm::sufficient_track_angle(r,obs,xyz);
}

unsigned merge(sfm::Reconstruction&r,Graph&g,std::ostream&log){
 g.refresh(r);std::map<std::pair<unsigned,unsigned>,std::vector<Edge>> pairs;
 for(auto e:g.edges){auto a=g.owner[e[0]],b=g.owner[e[1]];if(a==none||b==none||a==b)continue;if(a>b){std::swap(a,b);std::swap(e[0],e[1]);}pairs[{a,b}].push_back(e);}
 struct Candidate{unsigned support,a,b;};std::vector<Candidate> order;
 for(const auto&[p,es]:pairs){std::set<unsigned> left,right,cams;for(auto e:es){left.insert(e[0]);right.insert(e[1]);cams.insert(g.nodes[e[0]].camera);cams.insert(g.nodes[e[1]].camera);}if(es.size()>=2&&left.size()>=2&&right.size()>=2&&cams.size()>=3)order.push_back({unsigned(es.size()),p.first,p.second});}
 std::sort(order.begin(),order.end(),[](auto a,auto b){return std::tuple{-int(a.support),a.a,a.b}<std::tuple{-int(b.support),b.a,b.b};});
 std::set<unsigned> touched;unsigned accepted=0,conflicts=0,geometry_rejected=0; // one merge per track per round: never reuse stale support
 for(auto c:order){if(touched.contains(c.a)||touched.contains(c.b))continue;auto&a=r.tracks.at(c.a);auto&b=r.tracks.at(c.b);if(!a.valid||!b.valid)continue;
  if(!compatible_cameras(a.obs,b.obs)){++conflicts;continue;}
  auto obs=a.obs;obs.insert(obs.end(),b.obs.begin(),b.obs.end());V xyz{};
  if(!fit_all(r,obs,sfm::intersect_track(r,obs),xyz)){++geometry_rejected;continue;}
  // Additional independent prediction: each pre-merge 3D point must explain
  // the other fragment, preventing a bad compromise from accepting two points.
  unsigned cross_good=0;for(auto o:a.obs)cross_good+=sfm::geometry_error(r,o,b.xyz)<=6;for(auto o:b.obs)cross_good+=sfm::geometry_error(r,o,a.xyz)<=6;
  if(cross_good<obs.size()){++geometry_rejected;continue;}
  for(auto&o:obs)o.track=c.a;a.obs=std::move(obs);a.xyz=xyz;b.valid=false;touched.insert(c.a);touched.insert(c.b);++accepted;
  log<<"{\"merge\":true,\"target\":"<<c.a<<",\"source\":"<<c.b<<",\"support\":"<<c.support<<",\"views\":"<<a.obs.size()<<"}\n";
 }
 log<<"{\"merge_summary\":true,\"candidates\":"<<order.size()<<",\"accepted\":"<<accepted<<",\"camera_conflicts\":"<<conflicts<<",\"geometry_rejected\":"<<geometry_rejected<<"}\n";log.flush();g.refresh(r);return accepted;
}

unsigned complete(sfm::Reconstruction&r,Graph&g,std::ostream&log){
 g.refresh(r);struct Proposal{unsigned node,track,support;double error;};std::vector<Proposal> proposals;
 for(unsigned n=0;n<g.nodes.size();++n)if(g.owner[n]==none&&r.map.contains(g.nodes[n].camera)){
  std::map<unsigned,unsigned> targets;for(auto j:g.adjacent[n])if(g.owner[j]!=none)++targets[g.owner[j]];
  std::vector<Proposal> good;
  for(auto[id,support]:targets)if(support>=2){auto&t=r.tracks.at(id);if(std::any_of(t.obs.begin(),t.obs.end(),[&](auto o){return o.camera==g.nodes[n].camera;}))continue;double e=sfm::geometry_error(r,g.observation(n,id),t.xyz);if(e<=3)good.push_back({n,id,support,e});}
  std::sort(good.begin(),good.end(),[](auto a,auto b){return std::tuple{a.error,-int(a.support),a.track}<std::tuple{b.error,-int(b.support),b.track};});
  // Ambiguous assignments are left unassigned rather than made deterministic by ID.
  if(!good.empty()&&(good.size()==1||good[1].error-good[0].error>=1))proposals.push_back(good[0]);
 }
 std::sort(proposals.begin(),proposals.end(),[](auto a,auto b){return std::tuple{a.error,-int(a.support),a.node}<std::tuple{b.error,-int(b.support),b.node};});
 unsigned accepted=0;
 for(auto p:proposals){auto&t=r.tracks.at(p.track);if(std::any_of(t.obs.begin(),t.obs.end(),[&](auto o){return o.camera==g.nodes[p.node].camera;}))continue;auto obs=t.obs;obs.push_back(g.observation(p.node,p.track));V xyz{};
  if(!fit_all(r,obs,t.xyz,xyz))continue;t.obs=std::move(obs);t.xyz=xyz;g.owner[p.node]=p.track;++accepted;
  log<<"{\"completion\":true,\"track\":"<<p.track<<",\"node\":"<<p.node<<",\"support\":"<<p.support<<",\"prior_prediction_px\":"<<p.error<<"}\n";
 }
 log<<"{\"completion_summary\":true,\"proposals\":"<<proposals.size()<<",\"accepted\":"<<accepted<<"}\n";log.flush();g.refresh(r);return accepted;
}

unsigned tail(sfm::Reconstruction&r,Graph&g,std::ostream&log){
 // Predict a suspect observation from at least three OTHER views. This is
 // a training-time diagnostic, not the held-out evaluation metric.
 unsigned removed=0;
 for(auto&[id,t]:r.tracks)if(t.valid&&t.obs.size()>=4){
  for(unsigned pass=0;pass<2&&t.obs.size()>=4;++pass){
   auto worst=std::max_element(t.obs.begin(),t.obs.end(),[&](auto a,auto b){return sfm::geometry_error(r,a,t.xyz)<sfm::geometry_error(r,b,t.xyz);});
   double old_error=sfm::geometry_error(r,*worst,t.xyz);if(old_error<=3)break;auto dropped=*worst;auto obs=t.obs;obs.erase(obs.begin()+(worst-t.obs.begin()));
   V fitted=sfm::refine_track_point(r,obs,t.xyz);double predicted=sfm::geometry_error(r,dropped,fitted);double old_cost=sfm::point_cost(r,obs,t.xyz),new_cost=sfm::point_cost(r,obs,fitted);
   if(predicted<=4||new_cost>=old_cost||!sfm::sufficient_track_angle(r,obs,fitted))break;
   t.obs=std::move(obs);t.xyz=fitted;++removed;
   log<<"{\"tail_rejected\":true,\"track\":"<<id<<",\"camera\":"<<dropped.camera<<",\"u\":"<<dropped.u<<",\"v\":"<<dropped.v<<",\"fitted_error_px\":"<<old_error<<",\"other_views_prediction_px\":"<<predicted<<"}\n";
  }
 }
 g.refresh(r);log<<"{\"tail_summary\":true,\"removed_observations\":"<<removed<<"}\n";log.flush();return removed;
}

void exact_gate(sfm::Reconstruction&r,Graph&g,std::ostream&log){
 unsigned removed=0,invalid=0;
 for(auto&[id,t]:r.tracks)if(t.valid){for(;;){auto obs=t.obs;std::erase_if(obs,[&](auto o){return sfm::geometry_error(r,o,t.xyz)>4;});if(obs.size()==t.obs.size())break;removed+=unsigned(t.obs.size()-obs.size());t.obs=std::move(obs);if(t.obs.size()<2){t.valid=false;++invalid;break;}t.xyz=sfm::refine_track_point(r,t.obs,t.xyz);}if(t.valid&&!sfm::sufficient_track_angle(r,t.obs,t.xyz)){t.valid=false;++invalid;}}
 g.refresh(r);log<<"{\"final_gate\":true,\"removed\":"<<removed<<",\"invalid_points\":"<<invalid<<",\"gate_px\":4}\n";log.flush();
}

void save_model(sfm::Reconstruction&r,const fs::path&input,const fs::path&out){
 auto d=sfm::collect(r);sfm::configure(d,r.s);output(r.s,d,input,out);export_sparse(r.s,d,out/"sparse");write_array(out/"camera_ids.bin",r.ids);write_array(out/"active_points.bin",d.active);write_array(out/"observations.bin",d.observations);
 double sum=0,maximum=0;for(auto o:d.observations){double e=sfm::pixel_error(r.s.calibration,r.s.poses[o.camera],r.s.points[o.track],o);sum+=e*e;maximum=std::max(maximum,e);}
 std::ofstream(out/"summary.json")<<std::setprecision(17)<<"{\"aligned_cameras\":"<<r.ids.size()<<",\"sparse_points\":"<<d.active.size()<<",\"observations\":"<<d.observations.size()<<",\"reprojection_rmse_px\":"<<std::sqrt(sum/d.observations.size())<<",\"reprojection_max_px\":"<<maximum<<",\"fitted_intrinsics\":0,\"track_postprocessing_applied\":true,\"native_code_called\":false,\"absolute_accuracy_measured\":false}";
}

void self_test(){
 sfm::Reconstruction r;initial_calibration(r.s,3072,2048);V x{.1,.05,3};Graph g;
 for(unsigned cam=0;cam<6;++cam){Pose p;p.rotation=identity;set(p.position.data(),96,V{-.6+.24*cam,0,0});sfm::add(r,cam,p);sfm::Obs o{};o.camera=cam;o.size=1;auto v=evaluate(r.s.calibration.data(),p,x,o,false).residual;g.nodes.push_back({cam,0,float(-v[0]),float(-v[1])});}
 g.edges={{0,2},{1,3},{2,4},{3,4},{0,5}};g.initialize();
 auto add=[&](unsigned id,std::initializer_list<unsigned> nodes){auto&t=r.tracks[id];t.valid=true;t.xyz=x;for(auto n:nodes)t.obs.push_back(g.observation(n,id));};add(0,{0,1});add(1,{2,3});g.refresh(r);std::ostringstream log;
 if(merge(r,g,log)!=1||r.tracks[0].obs.size()!=4)throw std::runtime_error("synthetic graph merge");
 if(complete(r,g,log)!=1||r.tracks[0].obs.size()!=5||g.owner[5]!=none)throw std::runtime_error("multi-view completion support");
 if(compatible_cameras(r.tracks[0].obs,std::span<const sfm::Obs>(r.tracks[0].obs.data(),1)))throw std::runtime_error("camera conflict protection");
 auto bad=r.tracks[0];bad.obs[0].v+=10;bad.xyz=sfm::refine_track_point(r,bad.obs,x);V fitted{};
 if(fit_all(r,bad.obs,bad.xyz,fitted))throw std::runtime_error("outlier merge accepted");
 r.tracks[2]=r.tracks[0];bool rejected=false;try{g.refresh(r);}catch(const std::runtime_error&){rejected=true;}if(!rejected)throw std::runtime_error("duplicate ownership accepted");r.tracks.erase(2);g.refresh(r);
 std::cout<<"PASS: verified graph merge, two-view-supported completion, single-edge rejection, camera-conflict protection, geometric rejection, unique feature ownership\n";
}
}


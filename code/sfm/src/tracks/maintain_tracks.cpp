#define GRAPH6_NO_MAIN
#include "refine_graph.cpp"
#include "track_options.h"

// Track lifecycle refinement; engineering implementation informed by
// Schonberger/Frahm CVPR 2016 Sec.4.4. Not an implementation of PixSfM.
namespace life9 {
using graph6::Graph;using graph6::none;
struct Policy {double rms=2.5,predict=6;unsigned rounds=3;};

// A track that already survived geometric validation need not be connected
// in the graph induced by its remaining observations: a removed bridge is
// not evidence that its two valid sides are different physical points.
bool fit(const sfm::Reconstruction&r,std::vector<sfm::Obs>&obs,V seed,V&x,const Policy&p,bool predictive=true){
 std::set<unsigned> cameras;for(auto o:obs)if(!cameras.insert(o.camera).second)return false;
 if(obs.size()<2||!graph6::fit_all(r,obs,seed,x,4,p.rms))return false;
 if(predictive&&obs.size()>=3){unsigned tested=0;
  for(unsigned i=0;i<obs.size();++i){auto anchors=obs;anchors.erase(anchors.begin()+i);
   if(!sfm::sufficient_track_angle(r,anchors,x))continue;
   V q=sfm::refine_track_point(r,anchors,x);if(sfm::geometry_error(r,obs[i],q)>p.predict)return false;++tested;
  }if(tested<2)return false;
 }return true;
}

struct Merge {unsigned a,b,support;double gain;V x;std::vector<sfm::Obs> obs;};
unsigned merge(sfm::Reconstruction&r,Graph&g,const Policy&p,std::ostream&log){
 g.refresh(r);std::map<std::pair<unsigned,unsigned>,unsigned> pairs;
 for(auto[a,b]:g.edges){a=g.owner[a];b=g.owner[b];if(a!=none&&b!=none&&a!=b)++pairs[std::minmax(a,b)];}
 std::vector<Merge> candidates;unsigned conflicts=0,rejected=0;
 for(auto[key,support]:pairs){auto[a,b]=key;const auto&u=r.tracks.at(a);const auto&v=r.tracks.at(b);
  if(!graph6::compatible_cameras(u.obs,v.obs)){++conflicts;continue;}
  // A single verified link is sufficient to propose, never sufficient to accept.
  // All views and leave-one-view predictions must support the common point.
  auto obs=u.obs;obs.insert(obs.end(),v.obs.begin(),v.obs.end());V x{};
  V seed{};for(unsigned k=0;k<3;++k)seed[k]=(u.xyz[k]*u.obs.size()+v.xyz[k]*v.obs.size())/obs.size();
  if(!fit(r,obs,seed,x,p)){++rejected;continue;}
  double before=sfm::point_cost(r,u.obs,u.xyz)+sfm::point_cost(r,v.obs,v.xyz),after=sfm::point_cost(r,obs,x);
  // Unit-pixel model-complexity budget, a stated engineering assumption.
  double gain=3*std::log(2.0*obs.size())-(after-before);if(!(gain>0)){++rejected;continue;}
  candidates.push_back({a,b,support,gain,x,std::move(obs)});
 }
 std::sort(candidates.begin(),candidates.end(),[](const auto&a,const auto&b){return std::tuple{-a.gain,-int(a.support),a.a,a.b}<std::tuple{-b.gain,-int(b.support),b.a,b.b};});
 std::set<unsigned> touched;unsigned accepted=0;
 for(auto&c:candidates){if(touched.contains(c.a)||touched.contains(c.b))continue;
  for(auto&o:c.obs)o.track=c.a;r.tracks.at(c.a)={std::move(c.obs),c.x,true};r.tracks.at(c.b).valid=false;
  touched.insert(c.a);touched.insert(c.b);++accepted;
 }
 g.refresh(r);log<<"{\"lifecycle_merge\":true,\"candidate_track_pairs\":"<<pairs.size()<<",\"camera_conflicts\":"<<conflicts<<",\"geometry_or_model_rejected\":"<<rejected<<",\"valid_candidates\":"<<candidates.size()<<",\"accepted\":"<<accepted<<",\"observations_deleted\":0}\n";log.flush();return accepted;
}

unsigned complete(sfm::Reconstruction&r,Graph&g,const Policy&p,std::ostream&log){
 g.refresh(r);struct Add{unsigned node,id,support;double prediction;};std::vector<Add> proposals;unsigned ambiguous=0;
 for(unsigned n=0;n<g.nodes.size();++n)if(g.owner[n]==none&&r.map.contains(g.nodes[n].camera)){
  std::map<unsigned,unsigned> supports;for(auto j:g.adjacent[n])if(g.owner[j]!=none)++supports[g.owner[j]];
  std::vector<Add> candidates;
  for(auto[id,support]:supports){const auto&t=r.tracks.at(id);if(t.obs.size()<3||std::any_of(t.obs.begin(),t.obs.end(),[&](auto o){return o.camera==g.nodes[n].camera;}))continue;
   double e=sfm::geometry_error(r,g.observation(n,id),t.xyz);
   if(e<=(support>=2?3.0:2.0)&&(support>=2||t.obs.size()>=4))candidates.push_back({n,id,support,e});
  }
  std::sort(candidates.begin(),candidates.end(),[](auto a,auto b){return std::tuple{a.prediction,-int(a.support),a.id}<std::tuple{b.prediction,-int(b.support),b.id};});
  if(!candidates.empty()){if(candidates.size()==1||candidates[1].prediction-candidates[0].prediction>=1)proposals.push_back(candidates[0]);else ++ambiguous;}
 }
 std::sort(proposals.begin(),proposals.end(),[](auto a,auto b){return std::tuple{a.prediction,-int(a.support),a.node}<std::tuple{b.prediction,-int(b.support),b.node};});
 unsigned accepted=0,single=0;
 for(auto c:proposals){auto&t=r.tracks.at(c.id);if(g.owner[c.node]!=none||std::any_of(t.obs.begin(),t.obs.end(),[&](auto o){return o.camera==g.nodes[c.node].camera;}))continue;
  auto obs=t.obs;auto o=g.observation(c.node,c.id);double predicted=sfm::geometry_error(r,o,t.xyz);if(predicted>(c.support>=2?3:2))continue;
  obs.push_back(o);V x{};if(!fit(r,obs,t.xyz,x,p))continue;
  double increase=sfm::point_cost(r,obs,x)-sfm::point_cost(r,t.obs,t.xyz);if(increase>9)continue;
  t.obs=std::move(obs);t.xyz=x;g.owner[c.node]=c.id;++accepted;single+=c.support==1;
 }
 g.refresh(r);log<<"{\"lifecycle_completion\":true,\"proposed\":"<<proposals.size()<<",\"ambiguous_deferred\":"<<ambiguous<<",\"accepted\":"<<accepted<<",\"single_link_with_stricter_prediction\":"<<single<<",\"observations_deleted\":0}\n";log.flush();return accepted;
}

void tests(){
 sfm::Reconstruction r;initial_calibration(r.s,3072,2048);Graph g;V x{.1,.05,3};
 for(unsigned c=0;c<7;++c){Pose pose;pose.rotation=identity;set(pose.position.data(),96,V{-.6+.24*c,0,0});sfm::add(r,c,pose);sfm::Obs o{};o.size=1;auto uv=evaluate(r.s.calibration.data(),pose,x,o,false).residual;g.nodes.push_back({c,0,float(-uv[0]),float(-uv[1])});}
 g.nodes.push_back({6,1,g.nodes[6].u+60,g.nodes[6].v});g.edges={{0,1},{1,2},{2,3},{3,4},{5,6},{5,7}};g.initialize();
 auto make=[&](unsigned id,std::vector<unsigned> ns){auto&t=r.tracks[id];for(auto n:ns)t.obs.push_back(g.observation(n,id));t.xyz=x;t.valid=true;};
 make(0,{0,1,2});make(1,{3,4,5});g.refresh(r);Policy p;std::ostringstream log;
 if(merge(r,g,p,log)!=1)throw std::runtime_error("geometric whole-track merge failed");
 if(r.tracks.at(0).obs.size()!=6)throw std::runtime_error("historical disconnected endpoint was lost");
 if(complete(r,g,p,log)!=1||g.owner[6]==none||g.owner[7]!=none)throw std::runtime_error("completion or bad pending observation gate failed");
 auto conflict=r.tracks.at(0).obs;conflict.push_back(conflict[0]);V q;if(fit(r,conflict,x,q,p))throw std::runtime_error("camera duplicate accepted");
 g.edges.clear();g.initialize();g.refresh(r);if(merge(r,g,p,log)!=0)throw std::runtime_error("merge without verified edge accepted");
 auto d=sfm::collect(r);sfm::configure(d,r.s);calibrated::Policy bound;bound.observation_gate=4;
 if(!calibrated::feasible_observations(r.s,d,bound))throw std::runtime_error("valid BA state rejected");
 auto bad=r.s;bad.points[d.active.front()][0]+=1;
 if(calibrated::feasible_observations(bad,d,bound))throw std::runtime_error("BA residual constraint missing");
 std::cout<<"PASS V9 historical track retention, conservative merge, completion, camera uniqueness, graph evidence and bounded BA\n";
}
}

int wmain(int argc,wchar_t**argv){try{
 if(argc==2&&std::wstring(argv[1])==L"--self-test"){graph6::self_test();life9::tests();return 0;}
 if(argc<4||fs::exists(argv[2]))throw std::runtime_error("usage: maintain_tracks INPUT_SFM NEW_OUTPUT GRAPH [--no-track-recovery [--ba-rounds 0..3]]");
 auto options=TrackOptions::parse(argc,argv,true);
 fs::path input=argv[1],out=argv[2],graph=argv[3];
 auto r=calibrated::load_model(input);auto calibration=r.s.calibration;graph6::Graph g;g.nodes=read_array<graph6::Node>(graph/"nodes.bin");g.edges=read_array<graph6::Edge>(graph/"edges.bin");g.initialize();g.refresh(r);auto initial=g.owner;
 fs::create_directories(out);std::ofstream log(out/"optimization.jsonl");log<<std::setprecision(17);life9::Policy p;
 log<<"{\"track_recovery_enabled\":"<<(options.recovery?"true":"false")<<"}\n";
 for(unsigned round=0;round<(options.recovery?p.rounds:options.ba_rounds);++round){unsigned changed=0;
  if(options.recovery){
   for(unsigned sweep=0;sweep<5;++sweep){unsigned n=0;n+=life9::merge(r,g,p,log);n+=life9::complete(r,g,p,log);changed+=n;if(!n)break;}
   if(!changed)break;
  }
  log<<"{\"ba_call\":true,\"round\":"<<round<<",\"bounded\":true}\n";
  calibrated::Policy fixed;fixed.target=calibration;fixed.observation_gate=4;calibrated::refine(r,fixed,log);
  log<<"{\"lifecycle_round\":"<<round<<",\"accepted_operations\":"<<changed<<",\"ba_observation_gate_px\":4}\n";log.flush();
 }
 g.refresh(r);if(r.s.calibration!=calibration)throw std::runtime_error("fixed intrinsics changed");
 if(!options.recovery&&g.owner!=initial)throw std::runtime_error("disabled recovery changed track membership");
 std::vector<unsigned> pending;for(unsigned n=0;n<g.nodes.size();++n){if(initial[n]!=graph6::none&&g.owner[n]==graph6::none)throw std::runtime_error("input observation deleted");if(g.owner[n]==graph6::none)pending.push_back(n);}
 for(const auto&[id,t]:r.tracks)if(t.valid)for(auto o:t.obs)if(sfm::geometry_error(r,o,t.xyz)>4+1e-8)throw std::runtime_error("final residual feasibility failed");
 write_array(out/"pending_nodes.bin",pending);graph6::save_model(r,input,out);
 std::ofstream(out/"audit.json")<<"{\"passed\":true,\"unique_feature_ownership\":true,\"unique_track_camera\":true,\"fixed_intrinsics\":true,\"all_input_observations_preserved\":true,\"input_observations_deleted\":0,\"max_error_gate_px\":4,\"pending_count\":"<<pending.size()<<"}";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}

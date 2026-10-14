#define GRAPH6_NO_MAIN
#include "refine_graph.cpp"
#include <queue>
#include "track_options.h"

// Independent geometry-conditioned reconstruction from the complete verified
// feature graph. Old point IDs are initial hypotheses, never hard partitions.
namespace rebuild9 {
using graph6::Graph;
using graph6::none;
struct Policy { double rms=2.5, prediction=6; unsigned hops=5, rounds=3; };
struct Hypothesis { std::vector<unsigned> nodes; V xyz{}; double cost{},utility{}; };

std::vector<sfm::Obs> observations(const Graph&g,std::span<const unsigned>nodes,unsigned id=0){
 std::vector<sfm::Obs> out;for(auto n:nodes)out.push_back(g.observation(n,id));return out;
}

std::vector<unsigned> connected(const Graph&g,std::vector<unsigned> nodes){
 std::set<unsigned> left(nodes.begin(),nodes.end());std::vector<unsigned> best;
 while(!left.empty()){
  std::vector<unsigned> part{*left.begin()};left.erase(part.front());
  for(unsigned i=0;i<part.size();++i)for(auto n:g.adjacent[part[i]])if(left.erase(n))part.push_back(n);
  std::sort(part.begin(),part.end());if(part.size()>best.size()||(part.size()==best.size()&&part<best))best=std::move(part);
 }return best;
}

// Connectivity witnesses are original verified-graph features. They must
// agree geometrically (<=4 px, positive depth) but need not be selected as
// active observations. This separates graph connectivity from camera uniqueness.
std::vector<unsigned> connected_witness(const sfm::Reconstruction&r,const Graph&g,std::vector<unsigned> nodes,V xyz,const std::set<unsigned>&pool){
 std::set<unsigned> selected(nodes.begin(),nodes.end()),allowed=selected;
 for(auto n:pool)if(sfm::geometry_error(r,g.observation(n,0),xyz)<=4)allowed.insert(n);
 std::vector<unsigned> best;
 while(!allowed.empty()){
  std::vector<unsigned> frontier{*allowed.begin()},part;allowed.erase(frontier.front());
  for(unsigned k=0;k<frontier.size();++k){auto n=frontier[k];if(selected.contains(n))part.push_back(n);for(auto j:g.adjacent[n])if(allowed.erase(j))frontier.push_back(j);}
  std::sort(part.begin(),part.end());if(part.size()>best.size()||(part.size()==best.size()&&part<best))best=std::move(part);
 }return best;
}

bool fit(const sfm::Reconstruction&r,const Graph&g,Hypothesis&h,const Policy&p,bool predictive,const std::set<unsigned>*witness=nullptr){
 std::sort(h.nodes.begin(),h.nodes.end());h.nodes.erase(std::unique(h.nodes.begin(),h.nodes.end()),h.nodes.end());
 std::set<unsigned> cameras;for(auto n:h.nodes)if(!cameras.insert(g.nodes[n].camera).second)return false;
 for(unsigned pass=0;pass<8&&h.nodes.size()>=3;++pass){
  h.nodes=witness?connected_witness(r,g,std::move(h.nodes),h.xyz,*witness):connected(g,std::move(h.nodes));if(h.nodes.size()<3)return false;
  auto obs=observations(g,h.nodes);h.xyz=sfm::refine_track_point(r,obs,h.xyz);
  double worst=0;unsigned index=0;h.cost=0;
  for(unsigned i=0;i<obs.size();++i){double e=sfm::geometry_error(r,obs[i],h.xyz);h.cost+=e*e;if(e>worst){worst=e;index=i;}}
  if(worst>4){h.nodes.erase(h.nodes.begin()+index);continue;}
  if(!sfm::sufficient_track_angle(r,obs,h.xyz)||h.cost>p.rms*p.rms*obs.size())return false;
  // Only evaluate a withheld view when its remaining anchors have usable
  // parallax. The test uses original pixels, not adjusted pseudo-observations.
  if(predictive){
   double bad=0;unsigned bad_index=0;
   for(unsigned i=0;i<obs.size();++i){auto rest=obs;rest.erase(rest.begin()+i);if(!sfm::sufficient_track_angle(r,rest,h.xyz))continue;
    V x=sfm::refine_track_point(r,rest,h.xyz);double e=sfm::geometry_error(r,obs[i],x);if(e>bad){bad=e;bad_index=i;}}
   if(bad>p.prediction){h.nodes.erase(h.nodes.begin()+bad_index);continue;}
  }
  // Truncated residual benefit minus the cost of introducing an XYZ point.
  // Unit-pixel noise is an engineering assumption, not a fitted uncertainty.
  h.utility=16*h.nodes.size()-h.cost-3*std::log(2.0*h.nodes.size());
  return std::isfinite(h.utility)&&h.utility>0;
 }return false;
}

Hypothesis grow(const sfm::Reconstruction&r,const Graph&g,std::span<const unsigned> seeds,V xyz,const Policy&p,const std::vector<unsigned char>*blocked=nullptr){
 std::set<unsigned> pool;for(auto n:seeds)if(!blocked||!(*blocked)[n])pool.insert(n);
 for(unsigned iteration=0;iteration<2;++iteration){
  std::vector<unsigned> frontier(pool.begin(),pool.end());
  for(unsigned hop=0;hop<p.hops&&!frontier.empty();++hop){std::vector<unsigned> next;
   for(auto n:frontier)for(auto j:g.adjacent[n])if(!pool.contains(j)&&(!blocked||!(*blocked)[j])&&sfm::geometry_error(r,g.observation(j,0),xyz)<=4.5){pool.insert(j);next.push_back(j);}
   if(pool.size()>512)return {};frontier=std::move(next);
  }
  std::map<unsigned,std::pair<double,unsigned>> choice;
  for(auto n:pool){double e=sfm::geometry_error(r,g.observation(n,0),xyz);if(e>4.5)continue;unsigned support=0;for(auto j:g.adjacent[n])support+=pool.contains(j);
   auto value=std::pair{e*e-.25*std::min(support,4u),n};auto camera=g.nodes[n].camera;if(!choice.contains(camera)||value<choice[camera])choice[camera]=value;
  }
  Hypothesis h;h.xyz=xyz;for(auto[c,v]:choice)h.nodes.push_back(v.second);
  if(!fit(r,g,h,p,iteration==1,&pool))return {};xyz=h.xyz;
  if(iteration==1)return h;
 }return {};
}

struct Prefer {
 bool operator()(const Hypothesis&a,const Hypothesis&b)const{
  if(a.utility!=b.utility)return a.utility<b.utility;
  return a.nodes>b.nodes;
 }
};

unsigned allocate(const sfm::Reconstruction&r,const Graph&g,std::vector<Hypothesis> hypotheses,const Policy&p,std::vector<unsigned char>&used,std::vector<Hypothesis>&accepted){
 std::priority_queue<Hypothesis,std::vector<Hypothesis>,Prefer> queue;
 std::set<std::vector<unsigned>> unique;
 for(auto&h:hypotheses)if(h.nodes.size()>=3&&unique.insert(h.nodes).second)queue.push(std::move(h));
 unsigned added=0;
 while(!queue.empty()){
  auto h=queue.top();queue.pop();auto previous=h.nodes.size();std::erase_if(h.nodes,[&](auto n){return bool(used[n]);});
  if(h.nodes.size()<3)continue;
  if(h.nodes.size()!=previous){auto next=grow(r,g,h.nodes,h.xyz,p,&used);if(next.nodes.size()>=3)queue.push(std::move(next));continue;}
  for(auto n:h.nodes){if(used[n])throw std::runtime_error("allocation duplicate ownership");used[n]=1;}
  accepted.push_back(std::move(h));++added;
 }return added;
}

void rebuild(sfm::Reconstruction&r,Graph&g,const Policy&p,std::ostream&log,unsigned round){
 g.refresh(r);std::vector<Hypothesis> proposals;
 for(const auto&[id,t]:r.tracks)if(t.valid){auto h=grow(r,g,g.members.at(id),t.xyz,p);if(h.nodes.size()>=3)proposals.push_back(std::move(h));}
 unsigned proposed=unsigned(proposals.size());
 // Old points can sit on opposite sides of the same physical point. Seed
 // additional hypotheses from cross-track verified edges, rather than only
 // growing around each old XYZ. Keep two wide-baseline choices per old track.
 struct CrossSeed { double angle;graph6::Edge edge; };
 std::map<unsigned,std::vector<CrossSeed>> cross_seeds;
 for(auto edge:g.edges){auto[a,b]=edge;auto first=g.owner[a],second=g.owner[b];if(first==none||second==none||first==second)continue;
  const auto&ta=r.tracks.at(first);const auto&tb=r.tracks.at(second);auto oa=g.observation(a,0),ob=g.observation(b,0);
  if(sfm::geometry_error(r,oa,tb.xyz)>8||sfm::geometry_error(r,ob,ta.xyz)>8)continue;
  V middle{};for(unsigned k=0;k<3;++k)middle[k]=.5*(ta.xyz[k]+tb.xyz[k]);
  auto va=normalized(sub(middle,center(r.s.poses[r.map.at(oa.camera)]))),vb=normalized(sub(middle,center(r.s.poses[r.map.at(ob.camera)])));
  double angle=std::acos(std::clamp(std::abs(std::inner_product(va.begin(),va.end(),vb.begin(),0.0)),0.0,1.0));if(!(angle>.0261799388))continue;
  for(auto id:{first,second}){auto&v=cross_seeds[id];v.push_back({angle,edge});std::sort(v.begin(),v.end(),[](auto&a,auto&b){return std::pair{-a.angle,a.edge}<std::pair{-b.angle,b.edge};});if(v.size()>2)v.resize(2);}
 }
 std::set<graph6::Edge> attempted;unsigned cross_proposed=0;
 for(const auto&[id,seeds]:cross_seeds)for(const auto&s:seeds)if(attempted.insert(s.edge).second){auto obs=observations(g,s.edge);V x=sfm::intersect_track(r,obs);if(sfm::geometry_error(r,obs[0],x)>4||sfm::geometry_error(r,obs[1],x)>4)continue;auto h=grow(r,g,s.edge,x,p);if(h.nodes.size()>=3){proposals.push_back(std::move(h));++cross_proposed;}}
 std::vector<unsigned char> used(g.nodes.size());std::vector<Hypothesis> accepted;
 allocate(r,g,std::move(proposals),p,used,accepted);unsigned seeded=unsigned(accepted.size());
 // Deferred observations are never erased from g. Re-triangulate a limited
 // number of wide-baseline verified edges per free feature after allocation.
 unsigned pair_trials=0;
 for(unsigned pass=0;pass<2;++pass){std::vector<Hypothesis> recovered;std::set<graph6::Edge> tried;
  for(unsigned n=0;n<g.nodes.size();++n)if(!used[n]){
   struct Candidate{double angle;unsigned other;};std::vector<Candidate> neighbors;
   auto o=g.observation(n,0);auto a=unproject(r.s.calibration,{o.u,o.v});const auto&R=r.s.poses[r.map.at(o.camera)].rotation;a=normalized(V{R[0]*a[0]+R[3]*a[1]+R[6]*a[2],R[1]*a[0]+R[4]*a[1]+R[7]*a[2],R[2]*a[0]+R[5]*a[1]+R[8]*a[2]});
   for(auto j:g.adjacent[n])if(!used[j]){auto q=g.observation(j,0);auto b=unproject(r.s.calibration,{q.u,q.v});const auto&S=r.s.poses[r.map.at(q.camera)].rotation;b=normalized(V{S[0]*b[0]+S[3]*b[1]+S[6]*b[2],S[1]*b[0]+S[4]*b[1]+S[7]*b[2],S[2]*b[0]+S[5]*b[1]+S[8]*b[2]});double angle=std::acos(std::clamp(std::abs(std::inner_product(a.begin(),a.end(),b.begin(),0.0)),0.0,1.0));if(angle>.0261799388)neighbors.push_back({angle,j});}
   std::sort(neighbors.begin(),neighbors.end(),[](auto a,auto b){return std::pair{-a.angle,a.other}<std::pair{-b.angle,b.other};});
   for(unsigned k=0;k<std::min(2u,unsigned(neighbors.size()));++k){unsigned j=neighbors[k].other;graph6::Edge edge{std::min(n,j),std::max(n,j)};if(!tried.insert(edge).second)continue;++pair_trials;
    auto obs=observations(g,edge);auto x=sfm::intersect_track(r,obs);if(sfm::geometry_error(r,obs[0],x)>4||sfm::geometry_error(r,obs[1],x)>4)continue;
    auto h=grow(r,g,edge,x,p,&used);if(h.nodes.size()>=3)recovered.push_back(std::move(h));
   }
  }
  if(!allocate(r,g,std::move(recovered),p,used,accepted))break;
 }
 // Retain geometrically valid pre-existing two-view tracks when neither
 // feature was absorbed by a multi-view hypothesis. Do not create new
 // two-view fragments solely to fit unmatched observations.
 unsigned pairs=0;std::vector<Hypothesis> pair_hypotheses;
 for(const auto&[id,t]:r.tracks)if(t.valid&&t.obs.size()==2){auto ns=g.members.at(id);if(used[ns[0]]||used[ns[1]])continue;
  if(std::find(g.adjacent[ns[0]].begin(),g.adjacent[ns[0]].end(),ns[1])==g.adjacent[ns[0]].end())continue;
  auto obs=observations(g,ns);V x{};if(!graph6::fit_all(r,obs,t.xyz,x,4,p.rms))continue;
  pair_hypotheses.push_back({ns,x,sfm::point_cost(r,obs,x),0});
 }
 std::sort(pair_hypotheses.begin(),pair_hypotheses.end(),[](auto&a,auto&b){return std::pair{a.cost,a.nodes}<std::pair{b.cost,b.nodes};});
 for(auto&h:pair_hypotheses)if(!used[h.nodes[0]]&&!used[h.nodes[1]]){for(auto n:h.nodes)used[n]=1;accepted.push_back(std::move(h));++pairs;}
 r.tracks.clear();unsigned id=0;for(auto&h:accepted){r.tracks[id]={observations(g,h.nodes,id),h.xyz,true};++id;}
 if(r.tracks.size()<100)throw std::runtime_error("graph reconstruction retained too few tracks");g.refresh(r);
 auto active=std::count(used.begin(),used.end(),1);log<<"{\"graph_rebuild\":true,\"round\":"<<round<<",\"proposals_from_initial_points\":"<<proposed<<",\"cross_track_seed_trials\":"<<attempted.size()<<",\"cross_track_proposals\":"<<cross_proposed<<",\"accepted_seed_tracks\":"<<seeded<<",\"recovered_multiview_tracks\":"<<accepted.size()-seeded-pairs<<",\"retained_two_view_tracks\":"<<pairs<<",\"pair_seed_trials\":"<<pair_trials<<",\"points\":"<<r.tracks.size()<<",\"active_observations\":"<<active<<",\"pending_observations\":"<<g.nodes.size()-active<<"}\n";log.flush();
}

void filter_after_bundle(sfm::Reconstruction&r,Graph&g,const Policy&p,std::ostream&log){
 unsigned released=0;for(auto&[id,t]:r.tracks)if(t.valid){
  for(unsigned pass=0;pass<8;++pass){auto size=t.obs.size();std::erase_if(t.obs,[&](auto o){return sfm::geometry_error(r,o,t.xyz)>4;});released+=unsigned(size-t.obs.size());if(t.obs.size()<2){released+=unsigned(t.obs.size());t.valid=false;break;}t.xyz=sfm::refine_track_point(r,t.obs,t.xyz);if(std::all_of(t.obs.begin(),t.obs.end(),[&](auto o){return sfm::geometry_error(r,o,t.xyz)<=4;}))break;}
  if(t.valid&&std::any_of(t.obs.begin(),t.obs.end(),[&](auto o){return sfm::geometry_error(r,o,t.xyz)>4;})){released+=unsigned(t.obs.size());t.valid=false;}
  if(t.valid&&!sfm::sufficient_track_angle(r,t.obs,t.xyz)){released+=unsigned(t.obs.size());t.valid=false;}
 }g.refresh(r);log<<"{\"post_bundle_released_to_pending\":"<<released<<"}\n";log.flush();
}

void self_test(){
 sfm::Reconstruction r;initial_calibration(r.s,3072,2048);Graph g;V x{.1,.05,3};
 for(unsigned c=0;c<6;++c){Pose p;p.rotation=identity;set(p.position.data(),96,V{-.6+.24*c,0,0});sfm::add(r,c,p);sfm::Obs o{};o.size=1;auto q=evaluate(r.s.calibration.data(),p,x,o,false).residual;g.nodes.push_back({c,0,float(-q[0]),float(-q[1])});}
 g.nodes.push_back({0,1,g.nodes[0].u+40,g.nodes[0].v});g.edges={{0,1},{1,2},{2,3},{3,4},{4,5},{5,0},{6,2}};g.initialize();Policy p;
 std::array<unsigned,2> seed{0,1};auto h=grow(r,g,seed,x,p);if(h.nodes.size()!=6)throw std::runtime_error("raw graph did not recover six-view track");
 Hypothesis conflict{{0,6,1},x};if(fit(r,g,conflict,p,true))throw std::runtime_error("camera conflict accepted");
 std::vector<unsigned char> used(g.nodes.size());std::vector<Hypothesis> accepted;allocate(r,g,{h,h},p,used,accepted);if(accepted.size()!=1||std::count(used.begin(),used.end(),1)!=6||used[6])throw std::runtime_error("global ownership or pending-feature error");
 auto bridge=g.nodes[0];bridge.feature=2;bridge.u+=.1f;g.nodes.push_back(bridge);
 g.edges={{0,1},{1,2},{2,7},{7,3},{3,4},{4,5}};g.initialize();
 std::set<unsigned> witnesses{0,1,2,3,4,5,7};Hypothesis supported{{0,1,2,3,4,5},x};
 if(!fit(r,g,supported,p,true,&witnesses)||supported.nodes.size()!=6)throw std::runtime_error("valid same-camera connectivity witness lost");
 g.nodes[7].u+=100;g.initialize();supported={{0,1,2,3,4,5},x};
 if(fit(r,g,supported,p,true,&witnesses)&&supported.nodes.size()>3)throw std::runtime_error("geometric outlier used as bridge");
 g.edges.clear();g.initialize();h.nodes={0,1,2,3};if(fit(r,g,h,p,true))throw std::runtime_error("disconnected track accepted");
 std::cout<<"PASS final graph witness connectivity, rejected bad bridge, graph growth, conflict rejection, exclusive global assignment, pending-feature preservation and disconnected hypothesis rejection\n";
}
}

int wmain(int argc,wchar_t**argv){try{
 if(argc==2&&std::wstring(argv[1])==L"--self-test"){graph6::self_test();rebuild9::self_test();return 0;}
 if(argc<4||fs::exists(argv[2]))throw std::runtime_error("usage: rebuild_tracks INPUT_SFM NEW_OUTPUT FULL_GRAPH [--no-track-recovery]");
 auto options=TrackOptions::parse(argc,argv,false);
 fs::path input=argv[1],out=argv[2],graph=argv[3];rebuild9::Policy p;
 auto r=calibrated::load_model(input);auto calibration=r.s.calibration;graph6::Graph g;g.nodes=read_array<graph6::Node>(graph/"nodes.bin");g.edges=read_array<graph6::Edge>(graph/"edges.bin");g.initialize();g.refresh(r);
 auto initial=g.owner;
 fs::create_directories(out);std::ofstream log(out/"optimization.jsonl");log<<std::setprecision(17);
 log<<"{\"track_recovery_enabled\":"<<(options.recovery?"true":"false")<<"}\n";
 for(unsigned round=0;round<p.rounds;++round){
  if(options.recovery)rebuild9::rebuild(r,g,p,log,round);
  log<<"{\"ba_call\":true,\"round\":"<<round<<",\"bounded\":false}\n";
  calibrated::Policy fixed;fixed.target=calibration;calibrated::refine(r,fixed,log);rebuild9::filter_after_bundle(r,g,p,log);
 }
 if(r.s.calibration!=calibration)throw std::runtime_error("fixed calibration changed");g.refresh(r);
 if(!options.recovery)for(unsigned n=0;n<g.nodes.size();++n)if(g.owner[n]!=graph6::none&&g.owner[n]!=initial[n])throw std::runtime_error("disabled recovery added or reassigned an observation");
 std::vector<unsigned> pending;for(unsigned n=0;n<g.nodes.size();++n)if(g.owner[n]==graph6::none)pending.push_back(n);write_array(out/"pending_nodes.bin",pending);graph6::save_model(r,input,out);
 std::ofstream(out/"audit.json")<<"{\"passed\":true,\"unique_feature_ownership\":true,\"unique_track_camera\":true,\"fixed_intrinsics\":true,\"pending_features_retained_in_graph\":true,\"pending_count\":"<<pending.size()<<",\"track_recovery_enabled\":"<<(options.recovery?"true":"false")<<",\"surviving_input_point_partition_preserved\":"<<(options.recovery?"false":"true")<<"}";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}

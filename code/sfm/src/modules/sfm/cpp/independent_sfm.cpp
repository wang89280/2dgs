#ifdef MS_NATIVE_PARALLEL_COST
#undef MS_NATIVE_PARALLEL_COST
#endif
#include "candidate_cancellation.hpp"
#include "native_calibration_flow.hpp"
// Photos-derived observations and relative poses only. Original-engine files
// are consumed exclusively by the separate verification executables.
#define Observation PixelObservation
#include "../../matching/cpp/pose_inputs.hpp"
#undef Observation
#define wmain bundle_diagnostic_main
#include "../../calibration/cpp/bundle_adjust.cpp"
#undef wmain
#include "pnp_geometry.hpp"
#include "ray_triangulate.hpp"
#include <set>
#include <atomic>
#include <thread>
#include <mutex>
using photo_initial::Projection;
namespace sfm {
// Packaging adapter: preserve the 80-view algorithm, use the actual supplied count.
inline unsigned input_camera_count=80;
inline bool use_native_calibration_flow=false;
inline bool use_native_geometry_quality=false;
using Obs=bundle::Observation;
struct Track{std::vector<Obs>obs;V xyz{};bool valid{};};
struct Reconstruction{native_calibration::Stage calibration_stage{native_calibration::Stage::Incremental};bool full_calibration{},adaptive{true};State s;std::vector<unsigned>ids;std::map<unsigned,unsigned>map;std::map<unsigned,Track>tracks;};
Pose pose(const photo_initial::Camera& c){Pose p;p.rotation=c.rotation;set(p.position.data(),96,c.center);return p;}
photo_initial::Camera camera(const Pose&p){return {identity,p.rotation,center(p)};}
double pixel_error(const std::array<std::byte,256>&cal,const Pose&p,V x,const Obs&o){auto q=o;q.size=1;auto e=evaluate(cal.data(),p,x,q,false);return std::sqrt(2*e.cost);}
void configure(Data&d,const State&s){d.starts.clear();d.n=unsigned(d.parameters.size());for(auto&p:s.poses){d.starts.push_back(d.n);d.n+=position_dof(p)+(p.rotation_fixed?0:3);}d.by_point.assign(s.points.size(),{});for(unsigned i=0;i<d.observations.size();++i)d.by_point[d.observations[i].track].push_back(i);}
bool optimize(State&s,Data&d,unsigned iterations,std::ostream&log){configure(d,s);if(d.active.empty())return false;double lambda=0,current=cost(s,d);unsigned accepted=0;
 for(unsigned it=0;it<iterations;++it){candidate_checkpoint();try{auto n=assemble(s,d,lambda);auto delta=solve(n.schur);auto dp=back_substitute(n,delta);double direction=std::inner_product(delta.begin(),delta.end(),n.gradient.begin(),0.0);for(unsigned i=0;i<dp.size();++i)for(unsigned k=0;k<3;++k)direction+=dp[i][k]*n.points[i].gradient[k];double alpha=1,next=current;State candidate;while(alpha>1e-10){candidate=trial(s,d,delta,dp,alpha);next=cost(candidate,d);if(std::isfinite(next)&&current-alpha*.001*direction>=next)break;alpha*=.5;}log<<"{\"camera_count\":"<<s.poses.size()<<",\"points\":"<<d.active.size()<<",\"iteration\":"<<it<<",\"cost\":"<<current<<",\"candidate\":"<<next<<",\"alpha\":"<<alpha<<"}\n";log.flush();if(alpha<=1e-10||next>=current)break;s=std::move(candidate);current=next;++accepted;lambda/=10;double sum=0;for(double v:delta)sum+=v*v;double points=0;for(auto v:dp)for(double a:v)points+=a*a;if(std::sqrt(sum/d.n)*alpha<1e-6&&std::sqrt(points/(3*dp.size()))*alpha<1e-6)break;}catch(const std::runtime_error&e){lambda=lambda==0?.001:lambda*10;log<<"{\"linear_failure\":true,\"lambda\":"<<lambda<<"}\n";}}
 return accepted>0;
}
}
#include "pnp_refine.hpp"
namespace sfm {

struct PnpResult{Pose pose;std::vector<unsigned char>mask;unsigned inliers{};};
// Captured 142EBA990 -> 142467BD0: resection uses tolerance 0.0005 and
// ten trial steps. The bundle optimizer's 1e-6 tolerance is not used here.
PnpResult estimate_pose(const std::array<std::byte,256>&cal,std::span<const V>x,std::span<const Obs>obs,double threshold=10.240000486373901,bool refine=true){if(x.size()!=obs.size()||x.size()<4)throw std::runtime_error("PnP extent");std::vector<V>rays;for(auto o:obs)rays.push_back(unproject(cal,{o.u,o.v}));std::array<double,10>limits;limits[0]=threshold;for(unsigned i=1;i<10;++i)limits[i]=limits[i-1]*.75;std::array<unsigned,10>best{};std::array<Pose,10>poses;std::uint64_t random=1;int limit=500;
 for(int it=0;it<limit;++it){std::array<unsigned,3>ids;for(unsigned k=0;k<3;++k)for(;;){random=16807*random%2147483647;ids[k]=unsigned(random%x.size());if(std::find(ids.begin(),ids.begin()+k,ids[k])==ids.begin()+k)break;}std::array<V,3>w{x[ids[0]],x[ids[1]],x[ids[2]]},r{rays[ids[0]],rays[ids[1]],rays[ids[2]]};for(auto projection:photo_initial::p3p(w,r)){auto p=pose(photo_initial::decompose(projection));std::array<unsigned,10>count{};for(unsigned i=0;i<x.size();++i){double e=pixel_error(cal,p,x[i],obs[i]);for(unsigned l=0;l<10;++l)count[l]+=e<limits[l];}int proposed=-1;for(unsigned l=0;l<10;++l)if(count[l]>best[l]){best[l]=count[l];poses[l]=p;double ratio=double(count[l])/x.size();int next=0;if(ratio<1){next=limit; // 142EBA990 supplies confidence=1; 14246BE90 preserves the trial budget.
}proposed=std::max(proposed,next);}if(proposed>=0)limit=proposed;}}
 unsigned selected=0;if(best[0]>64)for(unsigned i=1;i<10;++i){if(best[i]*2<best[0])break;selected=i;}PnpResult result{poses[selected],std::vector<unsigned char>(x.size()),0};if(!best[selected])return result;for(unsigned i=0;i<x.size();++i){double e=pixel_error(cal,result.pose,x[i],obs[i]);result.mask[i]=e<limits[selected];}for(unsigned iteration=0;refine&&iteration<5;++iteration){result.pose=refine_pose(result.pose,cal,x,obs,result.mask,0.0005,10);unsigned changed=0;for(unsigned i=0;i<x.size();++i){bool yes=pixel_error(cal,result.pose,x[i],obs[i])<threshold;changed+=yes!=bool(result.mask[i]);result.mask[i]=yes;}if(!changed)break;}for(auto v:result.mask)result.inliers+=v;if(result.inliers<5){result.inliers=0;std::fill(result.mask.begin(),result.mask.end(),0);}return result;
}
void triangulate(Reconstruction&r,double threshold){
 threshold=threshold>0?std::min(threshold,5.120000243186951):5.120000243186951;
 for(auto&[id,t]:r.tracks){if(t.valid)continue;std::vector<photo_rays::Line>lines;std::vector<Obs>observations;
  for(auto o:t.obs)if(r.map.contains(o.camera)){auto&p=r.s.poses[r.map.at(o.camera)];auto d=unproject(r.s.calibration,{o.u,o.v});auto&R=p.rotation;V world{R[0]*d[0]+R[3]*d[1]+R[6]*d[2],R[1]*d[0]+R[4]*d[1]+R[7]*d[2],R[2]*d[0]+R[5]*d[1]+R[8]*d[2]};lines.push_back({center(p),world});observations.push_back(o);}
  if(lines.size()<2)continue;auto point=photo_rays::intersect(lines);bool valid=std::all_of(point.begin(),point.end(),[](double v){return std::isfinite(v);});
  for(auto o:observations){double weight=o.size?double(1.0f/o.size):1;valid&=pixel_error(r.s.calibration,r.s.poses[r.map.at(o.camera)],point,o)*weight<=threshold;}
  if(valid){t.xyz=point;t.valid=true;}
 }
}
Data collect(Reconstruction&r){Data d;d.parameters.clear(); // All external calibration parameters are fixed.
unsigned count=r.tracks.rbegin()->first+1;r.s.points.assign(count,{});for(auto&[id,t]:r.tracks)if(t.valid){unsigned observations=0;for(auto o:t.obs)observations+=r.map.contains(o.camera);if(observations<2)continue;r.s.points[id]=t.xyz;d.active.push_back(id);for(auto o:t.obs)if(r.map.contains(o.camera)){o.camera=r.map.at(o.camera);d.observations.push_back(o);}}return d;}
// Scene construction reorders active cameras and recreates the gauge for each
// optimizer stage. Verified against all captured stages, not a fixed seed pair.
void configure_gauge(Reconstruction&r){
 auto old=r.s.poses;auto ids=r.ids;std::sort(ids.begin(),ids.end());for(unsigned i=0;i<ids.size();++i)r.s.poses[i]=old[r.map.at(ids[i])];r.ids=ids;r.map.clear();for(unsigned i=0;i<ids.size();++i){r.map[ids[i]]=i;auto&p=r.s.poses[i];p.rotation_fixed=false;p.position[8]=p.position[9]=std::byte{};}
 if(ids.size()<2)throw std::runtime_error("gauge requires two cameras");auto&first=r.s.poses.front();auto&last=r.s.poses.back();first.rotation_fixed=true;first.position[8]=std::byte{1};last.position[9]=std::byte{1};auto origin=center(first);set(last.position.data(),16,origin);set(last.position.data(),40,norm(sub(center(last),origin)));auto direction=normalized(sub(center(last),origin));unsigned axis=std::abs(direction[0])>std::abs(direction[1]);if(std::abs(direction[axis])>std::abs(direction[2]))axis=2;V e{};e[axis]=1;auto b=normalized(cross(e,direction));set(last.position.data(),48,cross(direction,b));set(last.position.data(),72,b);
}
void bundle(Reconstruction&r,std::ostream&log,unsigned iterations=10){configure_gauge(r);auto d=collect(r);
 unsigned accepted_mask=65u,tentative_mask=r.full_calibration?3527u:193u;
 if(use_native_calibration_flow){auto policy=native_calibration::policy(r.calibration_stage);r.adaptive=policy.adaptive;accepted_mask=policy.accepted;tentative_mask=policy.tentative;iterations=policy.iterations;
  d.parameters.clear();for(auto bit:native_calibration::supported_bits)if(native_calibration::fitted_mask(policy)&(1u<<bit))d.parameters.push_back(bit);
  log<<"{\"calibration_stage\":\""<<policy.name<<"\",\"accepted_mask\":"<<accepted_mask<<",\"tentative_mask\":"<<tentative_mask<<",\"adaptive_prior\":"<<(r.adaptive?"true":"false")<<",\"iteration_limit\":"<<iterations<<"}\n";
 }
 // Native scene setup enables its internal adaptive prior during incremental
 // alignment even when public AlignCameras.adaptive_fitting is false.
 // Captured master masks: accepted f,k1 (65), tentative f,k1,k2 (193).
 d.parameters.clear();r.adaptive=false; // Preserve supplied principal point and distortion.
 if(r.adaptive){std::array<std::byte,904> master{};auto target=r.s.calibration;for(auto bit:fitted)if(!(accepted_mask&(1u<<bit)))set(target.data(),24+8*bit+(bit>=6?8:0),0.0);std::memcpy(master.data()+272,target.data(),256);master[848]=std::byte{1};set(master.data(),852,accepted_mask);set(master.data(),864,tentative_mask);set(master.data(),544,double(get<std::uint64_t>(target.data(),8)));set(master.data(),552,double(get<std::uint64_t>(target.data(),16)));d.priors=construct_prior(master.data(),true).priors;}
 optimize(r.s,d,iterations,log);for(auto id:d.active)r.tracks.at(id).xyz=r.s.points[id];
 // The adaptive optimizer fits tentative parameters, but scene publication
 // retains only the accepted mask. In every captured incremental stage of
 // this no-EXIF sensor the accepted mask is 65 (f,k1), including full fitting.
 // Do not feed tiny tentative k2/cx/cy/etc. updates into the next stage.
 if(r.adaptive)for(auto bit:fitted)if(!(accepted_mask&(1u<<bit)))set(r.s.calibration.data(),24+8*bit+(bit>=6?8:0),0.0);
 log<<"{\"stage_complete\":true,\"camera_count\":"<<r.ids.size()<<",\"f\":"<<get<double>(r.s.calibration.data(),24)<<",\"k1\":"<<get<double>(r.s.calibration.data(),80)<<",\"k2\":"<<get<double>(r.s.calibration.data(),88)<<",\"priors\":"<<d.priors.size()<<"}\n";log.flush();
}
void clean(Reconstruction&r,double threshold){for(auto&[id,t]:r.tracks)if(t.valid){unsigned bad=0,good=0;for(auto o:t.obs)if(r.map.contains(o.camera)){auto&p=r.s.poses[r.map.at(o.camera)];bool ok=mv(p.rotation,sub(t.xyz,center(p)))[2]>0&&pixel_error(r.s.calibration,p,t.xyz,o)<=threshold;good+=ok;bad+=!ok;}if(bad||good<2)t.valid=false;}}
// Native 142EA7530 and 142EB56E0 use a float reciprocal feature scale.
double weighted_error(const Reconstruction&r,const Track&t,const Obs&o){auto pixel=o;pixel.size=1;auto e=evaluate(r.s.calibration.data(),r.s.poses[r.map.at(o.camera)],t.xyz,pixel,false);double weight=o.size?double(1.0f/o.size):1.0;double u=e.residual[0]*weight,v=e.residual[1]*weight;return std::sqrt(v*v+u*u);}
// Frame-camera branches recovered from 142EB6660/142EB5BF0 and 142EB82A0.
#include "native_geometry_quality.hpp"
unsigned inaccurate(Reconstruction&r,double limit=double(1.0e-4f)){
 std::vector<std::pair<double,unsigned>>values;std::vector<double>scores;unsigned removed=0;
 for(auto&[id,t]:r.tracks)if(t.valid){auto q=geometry_quality(r,t);if(q.precision_risk){t.valid=false;++removed;continue;}values.push_back({q.score,id});scores.push_back(q.score);}
 double threshold=geometry_threshold(std::move(scores),limit);for(auto[value,id]:values)if(value<threshold){r.tracks.at(id).valid=false;++removed;}return removed;
}
unsigned weak(Reconstruction&r){if(r.ids.size()<3)return 0;std::vector<unsigned char>edges(r.ids.size()*r.ids.size());for(auto&[id,t]:r.tracks)if(t.valid){std::vector<unsigned>views;for(auto o:t.obs)if(r.map.contains(o.camera))views.push_back(r.map.at(o.camera));if(views.size()>2)for(auto a:views)for(auto b:views)edges[a*r.ids.size()+b]=1;}unsigned removed=0;for(auto&[id,t]:r.tracks)if(t.valid){std::vector<unsigned>views;for(auto o:t.obs)if(r.map.contains(o.camera))views.push_back(r.map.at(o.camera));if(views.size()==2&&views[0]!=views[1]&&!edges[views[0]*r.ids.size()+views[1]]){t.valid=false;++removed;}}return removed;}
unsigned filter_far(Reconstruction&r,double threshold){unsigned removed=0;threshold=threshold>0?std::min(threshold,5.120000243186951):5.120000243186951;for(auto&[id,t]:r.tracks)if(t.valid)for(auto o:t.obs)if(r.map.contains(o.camera)&&weighted_error(r,t,o)>threshold){t.valid=false;++removed;break;}return removed;}
// 142EB6A80, finite frame cameras: any aligned view with z <= 0
// invalidates the sparse point before the weak-track pass.
unsigned invisible(Reconstruction&r){unsigned removed=0;for(auto&[id,t]:r.tracks)if(t.valid)for(auto o:t.obs)if(r.map.contains(o.camera)){auto&p=r.s.poses[r.map.at(o.camera)];auto d=sub(t.xyz,center(p));double z=p.rotation[7]*d[1]+p.rotation[6]*d[0]+p.rotation[8]*d[2];if(z<=0){t.valid=false;++removed;break;}}return removed;}
unsigned filter_tracks(Reconstruction&r,double threshold,std::ostream&log){auto f=filter_far(r,threshold),a=inaccurate(r),v=invisible(r),w=weak(r);log<<"{\"sparse_far\":"<<f<<",\"sparse_inaccurate\":"<<a<<",\"sparse_invisible\":"<<v<<",\"sparse_weak\":"<<w<<"}\n";log.flush();return f+a+v+w;}
// 142E94B60 returns the signed NET change, not the number removed.
long long triangulate_filter(Reconstruction&r,double threshold,std::ostream&log){auto count=[&](){long long n=0;for(auto&[id,t]:r.tracks)n+=t.valid;return n;};auto before=count();triangulate(r,threshold);filter_tracks(r,threshold,log);return count()-before;}
// 142EB4CB0/142EB6F50, this data set's single finite frame sensor, no markers
// or external pose constraints. Removal cascades through under-observed points.
unsigned prune_unstable(Reconstruction&r,std::ostream&log){unsigned removed=0;
 for(;;){for(auto&[id,t]:r.tracks)if(t.valid){unsigned views=0;for(auto o:t.obs)views+=r.map.contains(o.camera);if(views<2)t.valid=false;}
  std::vector<unsigned>counts(r.ids.size());for(auto&[id,t]:r.tracks)if(t.valid)for(auto o:t.obs)if(r.map.contains(o.camera))++counts[r.map.at(o.camera)];
  std::vector<unsigned>keep;for(unsigned i=0;i<r.ids.size();++i)if(counts[i]>=3)keep.push_back(i);
  if(keep.size()==r.ids.size())break;auto ids=r.ids;auto poses=r.s.poses;removed+=unsigned(ids.size()-keep.size());r.ids.clear();r.map.clear();r.s.poses.clear();for(unsigned i:keep){r.map[ids[i]]=unsigned(r.ids.size());r.ids.push_back(ids[i]);r.s.poses.push_back(poses[i]);}
 }
 log<<"{\"unstable_cameras_removed\":"<<removed<<"}\n";log.flush();return removed;
}
// Captured 142E982A0: weak pass, then at most five ten-iteration bundles;
// stop as soon as the triangulation/filter manager adds fewer than one point.
void incremental_optimize(Reconstruction&r,std::ostream&log){weak(r);for(unsigned round=0;round<5;++round){prune_unstable(r,log);bundle(r,log,10);auto net=triangulate_filter(r,0,log);log<<"{\"incremental_round\":"<<round<<",\"net_sparse_change\":"<<net<<"}\n";log.flush();if(net<1)break;}}
void sigma_filter(Reconstruction&r,std::ostream&log){
 weak(r);for(unsigned round=0;round<5;++round){prune_unstable(r,log);bundle(r,log,20);double sum=0;std::size_t count=1;for(auto&[id,t]:r.tracks)if(t.valid){unsigned n=0;double local=0;for(auto o:t.obs)if(r.map.contains(o.camera)){double e=weighted_error(r,t,o);local+=e*e;++n;}if(n>1){sum+=local;count+=n;}}double sigma=std::sqrt(sum/count),threshold=std::min(5.120000243186951,3*sigma);unsigned removed=0;
  // As in 142E94B60, candidates are retriangulated before the far-point pass.
  triangulate(r,10.240000486373901);
  removed=filter_tracks(r,threshold,log);
  log<<"{\"sigma_filter_round\":"<<round<<",\"sigma\":"<<sigma<<",\"weighted_threshold\":"<<threshold<<",\"invalidated_sparse_points\":"<<removed<<"}\n";log.flush();
 }
}
void add(Reconstruction&r,unsigned id,Pose p){r.map[id]=unsigned(r.ids.size());r.ids.push_back(id);r.s.poses.push_back(p);}
void seed(Reconstruction&r,unsigned first,unsigned second,const fs::path&selected){
 // Native frame-camera initialization fixes the first camera to this world
 // orientation. Euler trial updates make the finite-iteration path gauge-sensitive.
 constexpr M world{1,0,0,0,-1,0,0,0,-1};Pose p;p.rotation=world;p.rotation_fixed=true;p.position[8]=std::byte{1};add(r,first,p);photo_initial::Camera b{identity,read_array<M>(selected/"refined_rotation.bin").at(0),read_array<V>(selected/"refined_center.bin").at(0)};b.rotation=mm(b.rotation,world);b.center=mv(world,b.center);p=pose(b);p.position[9]=std::byte{1};set(p.position.data(),40,norm(b.center));std::array<double,5>zero{};update(p,zero);add(r,second,p);triangulate(r,10.240000486373901);
}
unsigned resect(Reconstruction&r,std::ostream&events){
  struct Candidate{unsigned id,count;sfm::PnpResult result;std::vector<sfm::Obs> observations;};
  std::vector<Candidate> candidates;
  for(unsigned id=0;id<input_camera_count;++id)if(!r.map.contains(id)){
   std::vector<V>x;std::vector<sfm::Obs>o;
   for(auto&[track,t]:r.tracks)if(t.valid)for(auto observation:t.obs)if(observation.camera==id){x.push_back(t.xyz);o.push_back(observation);}
   if(x.size()<5)continue;
   auto result=sfm::estimate_pose(r.s.calibration,x,o);
   events<<"{\"candidate_camera\":"<<id<<",\"available\":"<<x.size()<<",\"inliers\":"<<result.inliers<<"}\n";
   candidates.push_back({id,unsigned(x.size()),std::move(result),std::move(o)});
  }
  events.flush();
  std::sort(candidates.begin(),candidates.end(),[](auto&a,auto&b){return std::pair{a.result.inliers,a.id}>std::pair{b.result.inliers,b.id};});
  if(candidates.empty()||candidates.front().result.inliers<5)return 0;
  unsigned best_count=candidates.front().result.inliers;
  // 142E92F90 computes this admission threshold from the best resection count.
  unsigned threshold=std::clamp(best_count>=100?best_count/2+25:3*best_count/4,2u,1000u),added=0;
  for(auto&c:candidates){if(c.result.inliers<threshold)break;sfm::add(r,c.id,c.result.pose);++added;
   // Preserve every observation. Native far-point filtering invalidates the point; // deleting the bad camera observation incorrectly rescues the whole track.
   unsigned rejected=0;for(auto keep:c.result.mask)rejected+=!keep;
   events<<"{\"accepted_camera\":"<<c.id<<",\"admission_threshold\":"<<threshold<<",\"pnp_outliers_retained_for_geometric_filtering\":"<<rejected<<"}\n";
  }
  return added;
}
#include "robust_tracks.hpp"
void finish(Reconstruction&r,const std::vector<Obs>&raw,std::ostream&log){
 r.full_calibration=true;sfm::sigma_filter(r,log);
 // Native group merge reloads the original photo observations before its
 // nonadaptive final sigma-filtering pass. These are our own photo tracks.
 // 142EAEFE3 calls the complete triangulation/filter manager after group
 // merge, before the first nonadaptive bundle. Triangulation alone admits
 // inaccurate or invisible points into that bundle and changes later filtering.
 r.tracks.clear();for(auto o:raw)r.tracks[o.track].obs.push_back(o);sfm::triangulate_filter(r,10.240000486373901,log);r.adaptive=false;sfm::sigma_filter(r,log);
 // Native alignment publishes this final sigma-filtered scene directly.
 // An additional 50-iteration bundle changes calibration after track cleanup.
}
struct InitialScore{unsigned aligned{},points{};double accuracy{},error{INFINITY};bool stable()const{return aligned==3&&points>=100&&accuracy>=.0025&&error<=.5;}};
// 142EA42A0 options +48..+60: 20 iterations, five rounds on each
// side of resection. Both loops stop when the manager's NET gain is <100.
bool initial_optimize(Reconstruction&r,std::ostream&log){if(use_native_calibration_flow)r.calibration_stage=native_calibration::Stage::Candidate;candidate_checkpoint();weak(r);for(unsigned round=0;round<5;++round){prune_unstable(r,log);if(r.ids.size()<2)return false;bundle(r,log,20);auto net=triangulate_filter(r,0,log);log<<"{\"initial_round\":"<<round<<",\"net_sparse_change\":"<<net<<"}\n";if(net<100)break;}return true;}
// 142EA42A0 evaluates a temporary three-camera reconstruction. Its optimized
// intrinsics/poses are discarded before production initialization.
InitialScore assess_initial(Reconstruction r,unsigned first,unsigned second,const fs::path&selected,std::ostream&log){
 candidate_checkpoint();r.adaptive=false;seed(r,first,second,selected);triangulate_filter(r,0,log);InitialScore score;
 if(!initial_optimize(r,log))return score;
 // The native pre-resection gate rejects a seed whose weighted camera RMSE
 // already exceeds options+40 (4.0), before trying to add a third camera.
 std::vector<double> seed_errors(r.ids.size());std::vector<unsigned> seed_counts(r.ids.size());
 for(auto&[id,t]:r.tracks)if(t.valid)for(auto o:t.obs)if(r.map.contains(o.camera)){auto k=r.map.at(o.camera);double e=weighted_error(r,t,o);seed_errors[k]+=e*e;++seed_counts[k];}
 for(unsigned k=0;k<seed_counts.size();++k)if(seed_counts[k]&&std::sqrt(seed_errors[k]/seed_counts[k])>4.0){score.aligned=unsigned(r.ids.size());return score;}
 unsigned best_id=UINT_MAX;PnpResult best;std::vector<Obs> best_observations;
 for(unsigned id=0;id<input_camera_count;++id)if(!r.map.contains(id)){
  std::vector<V>x;std::vector<Obs>o;for(auto&[track,t]:r.tracks)if(t.valid)for(auto obs:t.obs)if(obs.camera==id){x.push_back(t.xyz);o.push_back(obs);}
  if(x.size()<5)continue;auto p=estimate_pose(r.s.calibration,x,o);
  if(p.inliers>best.inliers){best_id=id;best=std::move(p);best_observations=std::move(o);}
 }
 if(best.inliers<5){score.aligned=unsigned(r.ids.size());return score;}
 add(r,best_id,best.pose);
 // PnP inliers select the pose; native scene observations remain intact for geometric filtering.
 triangulate_filter(r,0,log);if(!initial_optimize(r,log))return score;score.aligned=unsigned(r.ids.size());
 // 142EA5180 invalidates points with fewer than three aligned observations
 // before both accuracy and weighted camera-error evaluation (142EA59B0).
 for(auto&[id,t]:r.tracks)if(t.valid){unsigned views=0;for(auto o:t.obs)views+=r.map.contains(o.camera);if(views<3)t.valid=false;}
 std::array<double,3>errors{};std::array<unsigned,3>counts{};double accuracy=0;
 for(auto&[id,t]:r.tracks)if(t.valid){M h{};unsigned views=0;for(auto o:t.obs)if(r.map.contains(o.camera)){auto index=r.map.at(o.camera);auto e=evaluate(r.s.calibration.data(),r.s.poses[index],t.xyz,o,true,{});errors[index]+=2*e.cost;++counts[index];auto start=e.jacobian.size()-3;for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)h[i*3+j]+=o.size*o.size*(e.jacobian[start+i][0]*e.jacobian[start+j][0]+e.jacobian[start+i][1]*e.jacobian[start+j][1]);++views;}if(views==3){auto w=five::square_svd<3>(h).w;double length=std::sqrt(w[1]*w[1]+w[0]*w[0]+w[2]*w[2]);if(length>1e-15)accuracy+=std::min({w[0],w[1],w[2]})/length;++score.points;}}
 score.accuracy=score.points?accuracy/score.points:0;score.error=0;for(unsigned i=0;i<3;++i)score.error=std::max(score.error,counts[i]?std::sqrt(errors[i]/counts[i]):INFINITY);return score;
}
}

#ifndef MS_SFM_NO_MAIN
int wmain(int argc,wchar_t**argv){try{if(argc!=3||fs::exists(argv[2]))throw std::runtime_error("usage: incremental_sfm PHOTO_INITIAL NEW_OUTPUT");const bool robust=true,refine_cameras=true;fs::path input=argv[1],out=argv[2];auto raw=read_array<sfm::Obs>(input/"observations.bin");auto calibrations=read_array<std::array<std::byte,256>>(input/"calibrations.bin");if(calibrations.size()<3||calibrations.size()>80)throw std::runtime_error("expected 3 to 80 photo calibrations");sfm::input_camera_count=unsigned(calibrations.size());for(auto o:raw)if(o.camera>=sfm::input_camera_count)throw std::runtime_error("observation camera outside input calibration array");sfm::Reconstruction r;r.s.calibration=calibrations[0];for(auto o:raw)r.tracks[o.track].obs.push_back(o);fs::create_directories(out);std::ofstream log(out/"optimization.jsonl"),events(out/"events.jsonl");log<<std::setprecision(17);events<<std::setprecision(17);
 // Evaluate parallax using the captured 142E9E7A0 gate. Stability is tested by
 // actual incremental reconstruction below, not asserted from track counts.
 struct PairCandidate{unsigned score,common,first,second;fs::path directory;};std::vector<PairCandidate> pair_candidates;unsigned best=0,first=0,second=0;fs::path selected;for(auto&e:fs::directory_iterator(input))if(e.is_directory()&&e.path().filename().string().starts_with("pair_")){auto name=e.path().filename().string();unsigned a,b;if(sscanf_s(name.c_str(),"pair_%u_%u",&a,&b)!=2)continue;auto p=read_array<std::array<double,4>>(e.path()/"refined_triangulated_homogeneous.bin");auto c=read_array<V>(e.path()/"refined_center.bin").at(0);unsigned score=0;for(auto h:p){if(h[3]==0)continue;V x{h[0]/h[3],h[1]/h[3],h[2]/h[3]};auto u=normalized(x),v=normalized(sub(x,c));score+=std::abs(u[0]*v[0]+u[1]*v[1]+u[2]*v[2])<std::cos(2*3.141592653589793/180);}events<<"{\"candidate_first\":"<<a<<",\"second\":"<<b<<",\"parallax_score\":"<<score<<"}\n";pair_candidates.push_back({score,unsigned(fs::file_size(e.path()/"points_first.bin")/16),a,b,e.path()});if(score>best){best=score;first=a;second=b;selected=e.path();}}

 if(best<100)throw std::runtime_error("no nondegenerate initial pair");
 std::sort(pair_candidates.begin(),pair_candidates.end(),[](auto&a,auto&b){return initial_pair_pool_key(a.score,a.common,a.first,a.second)>initial_pair_pool_key(b.score,b.common,b.first,b.second);});
 std::ofstream stability(out/"initial_stability.jsonl"),stability_iterations(out/"initial_stability_iterations.jsonl");stability<<std::setprecision(17);stability_iterations<<std::setprecision(17);bool stable=false;
 auto record_score=[&](const PairCandidate&c,const sfm::InitialScore&score){stability<<"{\"first\":"<<c.first<<",\"second\":"<<c.second<<",\"aligned\":"<<score.aligned<<",\"three_view_points\":"<<score.points<<",\"accuracy\":"<<score.accuracy<<",\"max_camera_weighted_rmse\":"<<(std::isfinite(score.error)?score.error:0)<<",\"stable\":"<<(score.stable()?"true":"false")<<"}\n";stability.flush();};
 auto score_key=[](const sfm::InitialScore&s,const PairCandidate&c){return initial_candidate_key(s,c.first,c.second);};
 unsigned chosen=0;auto best_score=sfm::assess_initial(r,pair_candidates[0].first,pair_candidates[0].second,pair_candidates[0].directory,stability_iterations);record_score(pair_candidates[0],best_score);stable=best_score.stable();

 // Deterministic ranked-candidate policy: evaluate an ordered set, never
 // choose a winner based on worker completion time. This is our policy.
 const unsigned threshold=std::min(1000u,pair_candidates[0].score/2);
 unsigned tested=1;
 for(unsigned i=1;i<std::min(40u,unsigned(pair_candidates.size()));++i){
  auto&c=pair_candidates[i];if(c.score<threshold)continue;
  if(tested>=8&&stable)break;
  auto score=sfm::assess_initial(r,c.first,c.second,c.directory,stability_iterations);record_score(c,score);++tested;
  if(score.stable()&&(!stable||score_key(best_score,pair_candidates[chosen])<score_key(score,c))){chosen=i;best_score=score;stable=true;}
 }

 if(stable){auto&c=pair_candidates[chosen];first=c.first;second=c.second;selected=c.directory;}
 if(!stable)throw std::runtime_error("No evaluated initial pair passed native stability gates; expand photo pair candidates");
 sfm::seed(r,first,second,selected);sfm::triangulate_filter(r,0,log);sfm::incremental_optimize(r,log);
 while(r.ids.size()<sfm::input_camera_count){
  auto added=sfm::resect(r,events);if(!added)break;
  sfm::triangulate_filter(r,0,log);
  sfm::incremental_optimize(r,log);
  std::cout<<"aligned "<<r.ids.size()<<"/"<<sfm::input_camera_count<<", batch "<<added<<std::endl;
 }
 if(robust)sfm::robust_finish(r,raw,log,refine_cameras);else sfm::finish(r,raw,log);if(sfm::use_native_geometry_quality)sfm::write_geometry_quality(r,out/"geometry_quality.csv");auto d=sfm::collect(r);sfm::configure(d,r.s);auto oldposes=r.s.poses;std::vector<unsigned>order=r.ids;std::sort(order.begin(),order.end());std::map<unsigned,unsigned>remap;for(unsigned i=0;i<order.size();++i){remap[order[i]]=i;r.s.poses[i]=oldposes[r.map.at(order[i])];}for(auto&o:d.observations)o.camera=remap[r.ids[o.camera]];
 output(r.s,d,input,out);export_sparse(r.s,d,out/"sparse");write_array(out/"camera_ids.bin",order);write_array(out/"active_points.bin",d.active);write_array(out/"observations.bin",d.observations);double total=0,max_error=0;for(auto o:d.observations){double e=sfm::pixel_error(r.s.calibration,r.s.poses[o.camera],r.s.points[o.track],o);total+=e*e;max_error=std::max(max_error,e);}std::ofstream(out/"summary.json")<<std::setprecision(17)<<"{\"aligned_cameras\":"<<r.ids.size()<<",\"input_cameras\":"<<sfm::input_camera_count<<",\"sparse_points\":"<<d.active.size()<<",\"observations\":"<<d.observations.size()<<",\"reprojection_rmse_px\":"<<std::sqrt(total/d.observations.size())<<",\"reprojection_max_px\":"<<max_error<<",\"seed_first\":"<<first<<",\"seed_second\":"<<second<<",\"original_project_read\":false,\"native_code_called\":false,\"capture_input_read\":false,\"metric_scale_available\":false,\"native_initial_pair_stability_gates_applied\":true,\"native_additional_pair_search_fully_reproduced\":false,\"pnp_refinement_native_equivalence_verified\":false,\"full_cloud_accepted\":false}";return r.ids.size()==sfm::input_camera_count?0:3;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}

#endif

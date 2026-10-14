#pragma once
// Included inside namespace sfm after bundle(). This is an ALIKED integration
// improvement, not an assertion of equivalence to the commercial engine.
struct RobustTrackStats { unsigned points{}, observations{}, removed{}, short_baseline{}; };

inline std::array<double,2> point_residual(const Reconstruction& r,const Obs& observation,V xyz){
 auto pixel=observation;pixel.size=1;
 return evaluate(r.s.calibration.data(),r.s.poses[r.map.at(pixel.camera)],xyz,pixel,false).residual;
}
inline double point_cost(const Reconstruction&r,std::span<const Obs> observations,V xyz){
 double value=0;for(auto o:observations){auto e=point_residual(r,o,xyz);value+=e[0]*e[0]+e[1]*e[1];}return value;
}
// Numerically differentiate the actual forward residual, including its radius
// cap. The legacy analytic Jacobian has a different radial clamp in some domains.
inline V refine_track_point(const Reconstruction&r,std::span<const Obs> observations,V xyz){
 double current=point_cost(r,observations,xyz),lambda=1e-4;
 for(unsigned iteration=0;iteration<20&&std::isfinite(current);++iteration){
  M h{};V gradient{};
  for(auto o:observations){auto residual=point_residual(r,o,xyz);std::array<std::array<double,2>,3> jacobian{};
   for(unsigned k=0;k<3;++k){double step=1e-6*std::max(1.0,std::abs(xyz[k]));auto plus=xyz,minus=xyz;plus[k]+=step;minus[k]-=step;auto a=point_residual(r,o,plus),b=point_residual(r,o,minus);for(unsigned j=0;j<2;++j)jacobian[k][j]=(a[j]-b[j])/(2*step);}
   for(unsigned k=0;k<3;++k){for(unsigned j=0;j<2;++j)gradient[k]+=jacobian[k][j]*residual[j];for(unsigned l=0;l<3;++l)for(unsigned j=0;j<2;++j)h[k*3+l]+=jacobian[k][j]*jacobian[l][j];}
  }
  for(unsigned k=0;k<3;++k)h[k*3+k]+=lambda*std::max(h[k*3+k],1e-12);
  auto delta=mv(inverse3(h),gradient);auto next=sub(xyz,delta);double proposed=point_cost(r,observations,next);
  if(std::isfinite(proposed)&&proposed<current){double improvement=current-proposed;xyz=next;current=proposed;lambda=std::max(1e-12,lambda*.2);if(improvement<1e-10*std::max(1.0,current))break;}
  else {lambda*=10;if(lambda>1e10)break;}
 }
 return xyz;
}
inline V intersect_track(const Reconstruction&r,std::span<const Obs> observations){
 std::vector<photo_rays::Line> lines;
 for(auto o:observations){const auto&p=r.s.poses[r.map.at(o.camera)];auto ray=unproject(r.s.calibration,{o.u,o.v});const auto&R=p.rotation;
  lines.push_back({center(p),{R[0]*ray[0]+R[3]*ray[1]+R[6]*ray[2],R[1]*ray[0]+R[4]*ray[1]+R[7]*ray[2],R[2]*ray[0]+R[5]*ray[1]+R[8]*ray[2]}});
 }
 return photo_rays::intersect(lines);
}
inline bool sufficient_track_angle(const Reconstruction&r,std::span<const Obs> observations,V xyz){
 // Fold angles beyond 90 degrees; near-collinear opposite rays are also weak.
 constexpr double cosine_limit=0.9996573249755573; // cos(1.5 degrees)
 for(unsigned i=0;i<observations.size();++i){auto a=sub(xyz,center(r.s.poses[r.map.at(observations[i].camera)]));double na=norm(a);if(!(na>1e-12))continue;
  for(unsigned j=i+1;j<observations.size();++j){auto b=sub(xyz,center(r.s.poses[r.map.at(observations[j].camera)]));double nb=norm(b);if(!(nb>1e-12))continue;
   if(std::abs((a[0]*b[0]+a[1]*b[1]+a[2]*b[2])/(na*nb))<cosine_limit)return true;
  }
 }
 return false;
}

// V2 policy: all gates are Euclidean pixels; no detector-scale interpretation.
inline double geometry_error(const Reconstruction&r,const Obs&o,V x){
 if(!std::all_of(x.begin(),x.end(),[](double v){return std::isfinite(v);}))return INFINITY;
 const auto&p=r.s.poses[r.map.at(o.camera)];if(mv(p.rotation,sub(x,center(p)))[2]<=1e-10)return INFINITY;
 double e=pixel_error(r.s.calibration,p,x,o);return std::isfinite(e)?e:INFINITY;
}
inline bool recover_track(const Reconstruction&r,Track&t,RobustTrackStats&stats){
 constexpr double gate=4.0;
 std::vector<Obs> observations;std::set<unsigned> seen;
 for(auto o:t.obs)if(r.map.contains(o.camera)&&seen.insert(o.camera).second)observations.push_back(o);
 bool prior=t.valid;t.valid=false;if(observations.size()<2)return false;
 V best{};double best_cost=INFINITY;unsigned best_count=0;
 auto score=[&](V x){double cost=0;unsigned count=0;for(auto o:observations){double e=geometry_error(r,o,x);cost+=std::min(e*e,gate*gate);count+=e<=gate;}
  if(count>=2&&(count>best_count||(count==best_count&&cost<best_cost))){best=x;best_count=count;best_cost=cost;}};
 if(prior)score(t.xyz);score(intersect_track(r,observations));
 struct Pair{double angle;unsigned a,b;};std::vector<Pair>pairs;
 std::vector<V> rays;
 for(auto o:observations){auto d=unproject(r.s.calibration,{o.u,o.v});const auto&R=r.s.poses[r.map.at(o.camera)].rotation;rays.push_back(normalized(V{R[0]*d[0]+R[3]*d[1]+R[6]*d[2],R[1]*d[0]+R[4]*d[1]+R[7]*d[2],R[2]*d[0]+R[5]*d[1]+R[8]*d[2]}));}
 for(unsigned i=0;i<observations.size();++i)for(unsigned j=i+1;j<observations.size();++j){double dot=0;for(unsigned k=0;k<3;++k)dot+=rays[i][k]*rays[j][k];double angle=std::acos(std::clamp(std::abs(dot),0.0,1.0));if(angle>.0261799388)pairs.push_back({angle,i,j});}
 std::sort(pairs.begin(),pairs.end(),[](auto&a,auto&b){return std::tuple{a.angle,a.a,a.b}>std::tuple{b.angle,b.a,b.b};});
 unsigned trials=std::min(16u,unsigned(pairs.size()));
 for(unsigned k=0;k<trials;++k){unsigned ix=trials<=1?0:unsigned((std::uint64_t(k)*(pairs.size()-1))/(trials-1));auto q=pairs[ix];std::array<Obs,2>pair{observations[q.a],observations[q.b]};score(intersect_track(r,pair));}
 if(best_count<2)return false;
 std::vector<Obs>inliers;
 for(unsigned round=0;round<4;++round){inliers.clear();for(auto o:observations)if(geometry_error(r,o,best)<=gate)inliers.push_back(o);if(inliers.size()<2)return false;
  best=refine_track_point(r,inliers,best);
 }
 // Final fit and exact gate; delete observations, not their entire parent track.
 for(;;){inliers.clear();for(auto o:observations)if(geometry_error(r,o,best)<=gate)inliers.push_back(o);if(inliers.size()<2)return false;
  auto next=refine_track_point(r,inliers,best);std::vector<Obs>kept;for(auto o:inliers)if(geometry_error(r,o,next)<=gate)kept.push_back(o);best=next;
  if(kept.size()==inliers.size())break;observations=std::move(kept);
 }
 if(!sufficient_track_angle(r,inliers,best)){++stats.short_baseline;return false;}
 stats.removed+=unsigned(t.obs.size()-inliers.size());++stats.points;stats.observations+=unsigned(inliers.size());
 for(auto&o:inliers)o.size=1; t.obs=std::move(inliers);t.xyz=best;t.valid=true;return true;
}
inline void recover_all_tracks(Reconstruction&r,const std::vector<Obs>&raw,std::ostream&log,unsigned round){
 std::map<unsigned,Track> next;for(auto o:raw)next[o.track].obs.push_back(o);
 RobustTrackStats stats;unsigned splits=0,next_id=next.rbegin()->first+1;
 std::vector<std::pair<unsigned,Track>> additions;
 for(auto&[id,t]:next){auto old=r.tracks.find(id);if(old!=r.tracks.end()&&old->second.valid){t.xyz=old->second.xyz;t.valid=true;}
  auto original=t.obs;if(!recover_track(r,t,stats))continue;
  std::set<unsigned>used;for(auto o:t.obs)used.insert(o.camera);Track rest;
  for(auto o:original)if(!used.contains(o.camera))rest.obs.push_back(o);
  // A split needs three independent camera observations; never duplicate features.
  if(rest.obs.size()>=3){RobustTrackStats local;if(recover_track(r,rest,local)&&rest.obs.size()>=3){unsigned child=next_id++;for(auto&o:rest.obs)o.track=child;additions.push_back({child,std::move(rest)});++splits;stats.points+=local.points;stats.observations+=local.observations;}}
 }
 for(auto&[id,t]:additions)next.emplace(id,std::move(t));r.tracks=std::move(next);
 log<<"{\"v2_recovery_round\":"<<round<<",\"points\":"<<stats.points<<",\"observations\":"<<stats.observations<<",\"removed_observations_from_primary_tracks\":"<<stats.removed<<",\"split_tracks\":"<<splits<<",\"short_baseline\":"<<stats.short_baseline<<",\"gate_px\":4,\"angle_deg\":1.5}\n";log.flush();
 if(stats.points<100)throw std::runtime_error("V2 geometry retained fewer than 100 points");
}
inline double huber_objective(const State&s,const Data&d){double loss=0;for(auto o:d.observations){o.size=1;double e=pixel_error(s.calibration,s.poses[o.camera],s.points[o.track],o);loss+=e<=2?.5*e*e:2*(e-1);}return loss;}
inline void robust_bundle_v2(Reconstruction&r,std::ostream&log){
 configure_gauge(r);auto d=collect(r);configure(d,r.s);
 for(unsigned irls=0;irls<3;++irls){auto before=r.s;double old=huber_objective(r.s,d);unsigned downweighted=0;
  for(auto&o:d.observations){auto pixel=o;pixel.size=1;double e=pixel_error(r.s.calibration,r.s.poses[o.camera],r.s.points[o.track],pixel);double w=e<=2?1:2/e;o.size=float(1/std::sqrt(std::max(w,1e-6)));downweighted+=w<1;}
  optimize(r.s,d,15,log);double after=huber_objective(r.s,d);bool accepted=std::isfinite(after)&&after<=old+1e-8*std::max(1.0,old);
  if(!accepted)r.s=std::move(before);
  log<<"{\"v2_irls\":"<<irls<<",\"huber_before\":"<<old<<",\"huber_after\":"<<after<<",\"accepted\":"<<(accepted?"true":"false")<<",\"downweighted\":"<<downweighted<<"}\n";log.flush();
  if(!accepted||std::abs(old-after)<1e-6*std::max(1.0,old))break;
 }
 for(auto id:d.active)r.tracks.at(id).xyz=r.s.points[id];
}
inline void robust_finish(Reconstruction&r,const std::vector<Obs>&raw,std::ostream&log,bool refine_cameras){
 recover_all_tracks(r,raw,log,0);
 if(refine_cameras)for(unsigned round=1;round<=3;++round){robust_bundle_v2(r,log);recover_all_tracks(r,raw,log,round);}
}

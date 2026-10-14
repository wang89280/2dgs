#pragma once
// Included inside namespace sfm. Native frame-sensor quality criterion, verified
// against 114 captured filtering calls on the two masked image datasets.
struct PointGeometryQuality {double score{};bool precision_risk{};unsigned views{};};
inline PointGeometryQuality geometry_quality(const Reconstruction&r,const Track&t){
 M h{};bool bad=false;unsigned views=0;
 for(auto o:t.obs)if(r.map.contains(o.camera)){
  ++views;auto pixel=o;pixel.size=1;auto e=evaluate(r.s.calibration.data(),r.s.poses[r.map.at(o.camera)],t.xyz,pixel,true,{});auto start=e.jacobian.size()-3;double ps=point_scale(t.xyz),length=0;
  for(unsigned i=0;i<3;++i){for(unsigned j=0;j<3;++j)h[i*3+j]+=(e.jacobian[start+i][0]*e.jacobian[start+j][0]+e.jacobian[start+i][1]*e.jacobian[start+j][1])/(ps*ps);length+=(e.jacobian[start+i][0]*e.jacobian[start+i][0]+e.jacobian[start+i][1]*e.jacobian[start+i][1])/(ps*ps);}
  bad|=norm(t.xyz)*std::sqrt(length)*0x1p-23>(o.size?double(o.size):1.0);
 }
 auto w=five::square_svd<3>(h).w;double length=std::sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);
 return {length>=1e-20?std::min({w[0],w[1],w[2]})/length:0,bad,views};
}
inline double geometry_threshold(std::vector<double> scores,double limit=double(1.0e-4f)){
 if(scores.empty())return 1e-6;std::sort(scores.begin(),scores.end());return std::max(1e-6,std::min(limit,scores[scores.size()/2]*.1));
}
inline void write_geometry_quality(const Reconstruction&r,const fs::path&path){
 std::ofstream f(path);f<<std::setprecision(17)<<"track,views,hessian_condition_score,float_precision_risk\n";
 for(auto&[id,t]:r.tracks)if(t.valid){auto q=geometry_quality(r,t);f<<id<<','<<q.views<<','<<q.score<<','<<q.precision_risk<<'\n';}
 if(!f)throw std::runtime_error("geometry quality audit write failed");
}

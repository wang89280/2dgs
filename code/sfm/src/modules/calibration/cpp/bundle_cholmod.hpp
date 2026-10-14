#pragma once
#include <cholmod.h>
inline std::vector<double> solve(const System&s){
 cholmod_common c;cholmod_start(&c);cholmod_sparse*a=nullptr;cholmod_dense*b=nullptr,*x=nullptr;cholmod_factor*l=nullptr;
 auto release=[&](){cholmod_free_dense(&x,&c);cholmod_free_dense(&b,&c);cholmod_free_factor(&l,&c);cholmod_free_sparse(&a,&c);cholmod_finish(&c);};
 try{std::size_t nnz=0;for(std::size_t col=0;col<s.n;++col)for(std::size_t row=col;row<s.n;++row)nnz+=s.pattern.empty()?(row==col||s.a[row*s.n+col]!=0):s.pattern[row*s.n+col]!=0;
  a=cholmod_allocate_sparse(s.n,s.n,nnz,1,1,-1,CHOLMOD_REAL,&c);b=cholmod_allocate_dense(s.n,1,s.n,CHOLMOD_REAL,&c);if(!a||!b)throw std::runtime_error("CHOLMOD allocation");
  auto ap=static_cast<int*>(a->p),ai=static_cast<int*>(a->i);auto ax=static_cast<double*>(a->x);std::size_t k=0;for(std::size_t col=0;col<s.n;++col){ap[col]=int(k);for(std::size_t row=col;row<s.n;++row)if(s.pattern.empty()?(row==col||s.a[row*s.n+col]!=0):s.pattern[row*s.n+col]!=0){ai[k]=int(row);ax[k++]=s.a[col*s.n+row];}}ap[s.n]=int(k);std::copy(s.b.begin(),s.b.end(),static_cast<double*>(b->x));
  l=cholmod_analyze(a,&c);c.quick_return_if_not_posdef=1;if(!l||!cholmod_factorize(a,l,&c)||c.status!=CHOLMOD_OK)throw std::runtime_error("CHOLMOD factorization");x=cholmod_solve(CHOLMOD_A,l,b,&c);if(!x||c.status!=CHOLMOD_OK)throw std::runtime_error("CHOLMOD solve");auto values=static_cast<double*>(x->x);std::vector<double> result(values,values+s.n);release();return result;
 }catch(...){release();throw;}
}

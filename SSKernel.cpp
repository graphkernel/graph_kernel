// single_source.cpp  –  Struct60, ε=0.1, δ=0.01
// ------------------------------------------------
#include <bits/stdc++.h>
#include <filesystem>
using namespace std;  namespace fs = std::filesystem;

/* ======= global settings ======= */
constexpr double EPS  = 0.10;
constexpr double DELTA= 0.01;
constexpr double ALPHA= 0.15;
constexpr int    L_DET= 5;          // deterministic prefix
constexpr uint32_t SEED = 123;

/* ======= Split struct (fixed) === */
struct Split { double et, er, ew; };
const Split SP = {0.20*EPS, 0.60*EPS, 0.20*EPS};

/* ======= Simple graph =========== */
struct Graph{
    int n; vector<vector<int>> adj;
    explicit Graph(int N=0): n(N), adj(N){}
    void add_edge(int u,int v){ adj[u].push_back(v); adj[v].push_back(u); }
};

/* read edgelist, 0-index */
Graph read_edge(const string& f){
    ifstream fin(f); if(!fin) throw;
    vector<pair<int,int>> E; vector<int> ids; string ln;
    while(getline(fin,ln)){
        if(ln.empty()||ln[0]=='#') continue;
        istringstream is(ln); int u,v; if(is>>u>>v){E.emplace_back(u,v);ids.push_back(u);ids.push_back(v);}
    }
    sort(ids.begin(),ids.end()); ids.erase(unique(ids.begin(),ids.end()),ids.end());
    unordered_map<int,int> mp; for(int i=0;i<(int)ids.size();++i) mp[ids[i]]=i;
    Graph G(ids.size()); for(auto [u,v]:E) G.add_edge(mp[u],mp[v]);
    return G;
}

/* ======= deterministic prefix + walk sampler ======= */
vector<double> pre_rw(const Graph& G,int s,int K,long nW,mt19937& rng){
    const double p = sqrt(1-ALPHA);
    vector<double> RW((K+1)*G.n,0.0);

    /* prefix via mat-vec */
    vector<double> vec(G.n,0), nxt(G.n,0);
    vec[s]=1.0;
    for(int step=0; step<=min(L_DET,K); ++step){
        for(int v=0; v<G.n; ++v) RW[step*G.n+v]+=vec[v];       // record
        if(step==K) break;
        fill(nxt.begin(),nxt.end(),0.0);
        for(int u=0; u<G.n; ++u)
            if(!G.adj[u].empty()){
                double share = p/ G.adj[u].size();
                for(int v:G.adj[u]) nxt[v]+=share*vec[u];
                nxt[u]+= (1-p)*vec[u];
            }
        vec.swap(nxt);
    }

    /* remaining via sampling */
    uniform_real_distribution<> uni(0,1);
    for(long w=0; w<nW; ++w){
        int v=s, step=0;
        while(step<=K){
            if(step>=L_DET) RW[step*G.n+v]+=1.0;
            if(uni(rng)>p||G.adj[v].empty()) break;
            v=G.adj[v][uniform_int_distribution<>(0,(int)G.adj[v].size()-1)(rng)];
            ++step;
        }
    }
    for(double& x:RW) x/=nW;                    // average walks
    return RW;
}

/* ======= estimator (multi-target) ======= */
vector<double> estimate(
        const Graph& G,const Graph& H,
        int sG,int sH,int K,long nW,long T,
        const vector<pair<int,int>>& TGT,mt19937& rng)
{
    const int nG=G.n, nH=H.n, M=TGT.size();
    auto RG=pre_rw(G,sG,K,nW,rng), RH=pre_rw(H,sH,K,nW,rng);
    const double norm = ALPHA / pow(1 - sqrt(1-ALPHA),2);

    vector<int> tu(M), tv(M);
    for(int i=0;i<M;++i){ tu[i]=TGT[i].first; tv[i]=TGT[i].second; }

    uniform_int_distribution<> rad(0,1);
    vector<double> vG(nG), vH(nH), res(M,0.0);

    for(long t=0;t<T;++t){
        fill(vG.begin(),vG.end(),0); fill(vH.begin(),vH.end(),0);
        for(int k=0;k<=K;++k){
            int sgn=rad(rng)?1:-1;
            const double* pg=&RG[k*nG];
            const double* ph=&RH[k*nH];
            for(int u=0;u<nG;++u) vG[u]+=sgn*pg[u];
            for(int v=0;v<nH;++v) vH[v]+=sgn*ph[v];
        }
        for(int i=0;i<M;++i) res[i]+=vG[tu[i]]*vH[tv[i]];
    }
    for(double& x:res) x = norm * x / T;
    return res;
}
vector<pair<int,int>> sample_target_pairs(
        const Graph& G,const Graph& H,int k,mt19937& rng){
    uniform_int_distribution<> dG(0,G.n-1), dH(0,H.n-1);
    unordered_set<long long> seen; vector<pair<int,int>> out; out.reserve(k);
    while((int)out.size()<k){
        int u=dG(rng), v=dH(rng);
        long long key=1LL*u*H.n+v;
        if(seen.insert(key).second) out.emplace_back(u,v);
    }
    return out;
}
int main(){
    Graph G=read_edge("./graphs/facebook.txt"); 
    mt19937 rng(SEED);

    int  K  = ceil(log(SP.et)/log(1-ALPHA));             // truncation
    long T  = ceil( 3.0*K*log(4*(K+1)/DELTA) / (SP.er*SP.er) );
    long nW = ceil( pow(K+1,2)*log(4*(K+1)/DELTA) / (SP.ew*SP.ew) );

    int sG=0, sH=0;                                      
    auto targets = sample_target_pairs(G,G,5000,rng);

    auto est = estimate(G,G,sG,sH,K,nW,T,targets,rng);
    return 0;
}

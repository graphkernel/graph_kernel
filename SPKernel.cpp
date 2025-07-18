#include <bits/stdc++.h>
using namespace std;

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

/* ---------- compressed pre_rw (only target node) ---------- */
vector<double> pre_single(const Graph& G,int s,int target,int K,long nW,mt19937& rng){
    const double p=sqrt(1-ALPHA);
    vector<double> prob(K+1,0.0);

    /* deterministic prefix */
    double mass_on[target?2:1];              
    vector<double> cur(G.n,0), nxt(G.n,0);   
    cur[s]=1.0;
    for(int step=0; step<=min(L_DET,K); ++step){
        prob[step]+=cur[target];
        if(step==K) break;
        fill(nxt.begin(),nxt.end(),0.0);
        for(int u=0; u<G.n; ++u)
            if(!G.adj[u].empty()){
                double share=p/G.adj[u].size();
                for(int v:G.adj[u]) nxt[v]+=share*cur[u];
                nxt[u]+=(1-p)*cur[u];
            }
        cur.swap(nxt);
    }

    /* sampling */
    uniform_real_distribution<> uni(0,1);
    for(long w=0; w<nW; ++w){
        int v=s, step=0;
        while(step<=K){
            if(step>=L_DET && v==target) prob[step]+=1.0;
            if(uni(rng)>p||G.adj[v].empty()) break;
            v=G.adj[v][uniform_int_distribution<>(0,(int)G.adj[v].size()-1)(rng)];
            ++step;
        }
    }
    for(double& x:prob) x/=nW;
    return prob;
}

/* ---------- estimator (single-pair) ---------- */
double estimate_pair(const Graph& G,const Graph& H,
                     int sG,int sH,int u,int v,
                     int K,long nW,long T,mt19937& rng)
{
    auto RG = pre_single(G,sG,u,K,nW,rng);
    auto RH = pre_single(H,sH,v,K,nW,rng);
    uniform_int_distribution<> rad(0,1);
    double acc=0;
    for(long t=0;t<T;++t){
        double gSum=0,hSum=0;
        for(int k=0;k<=K;++k){
            int sgn=rad(rng)?1:-1;
            gSum+=sgn*RG[k];
            hSum+=sgn*RH[k];
        }
        acc+=gSum*hSum;
    }
    double norm = ALPHA / pow(1 - sqrt(1-ALPHA),2);
    return norm * acc / T;
}

/* ---------- main skeleton ---------- */
int main(){
    Graph G=read_edge("./graphs/facebook.txt");
    mt19937 rng(SEED);
    int sG=0,sH=0,u=1726,v=2351;            

    int  K  = ceil(log(SP.et)/log(1-ALPHA));
    long T  = ceil( 3.0*K*log(4*(K+1)/DELTA) / (SP.er*SP.er) );
    long nW = ceil( pow(K+1,2)*log(4*(K+1)/DELTA) / (SP.ew*SP.ew) );

    double est = estimate_pair(G,G,sG,sH,u,v,K,nW,T,rng);
    cout<<"sG "<<sG<<" sH "<<sH<<" → ("<<u<<","<<v<<") = "<<setprecision(17)<<est<<"\n";
    return 0;
}

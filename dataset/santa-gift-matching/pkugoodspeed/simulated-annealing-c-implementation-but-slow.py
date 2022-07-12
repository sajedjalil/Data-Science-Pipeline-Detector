# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/santa-gift-matching"]).decode("utf8"))
print(check_output(["ls", "../input/c-output"]).decode("utf8"))
import os

# Any results you write to the current directory are saved as output.

f = open('test.cpp', 'w')
f.write("""
#include<bits/stdc++.h>
#include<cassert>
using namespace std;

typedef long double LD;
typedef pair<LD, int> di;

const int NUM_CHLD = 1000000;
const int NUM_GIFT = 1000;
const int GIFT_LIMT = 1000;
const int WISH_SIZE = 100;
const int PREF_SIZE = 1000;
const int NUM_TRIP = 1667;
const int NUM_TWIN = 20000;
const int NUM_MINUTE = 55;

class SantaGifts{
    const LD score_factor = 2.0;
    const int triplet_lim = 5001;
    const int twin_lim = 45001;
    const int roll_lim = 200;
    vector<unordered_map<int, LD>> wish_score;
    vector<unordered_map<int, LD>> gift_score;
    
    LD res, current;
    LD sum_ch, sum_sh;
    vector<int> picks;
    vector<int> optimal;             // childIdx to giftIdx
    vector<vector<int> > assignment; // assignment of each color to a chidren list
    LD getChildWishScore(int childIdx, int giftIdx){
        if(wish_score[childIdx].count(giftIdx)) {
            return wish_score[childIdx][giftIdx];
        }
        return -1./LD(WISH_SIZE)/score_factor;
    }
    LD getSantaGiftScore(int childIdx, int giftIdx){
        if(gift_score[childIdx].count(giftIdx)){
            return wish_score[childIdx][giftIdx];
        }
        return -1./LD(PREF_SIZE)/score_factor;
    }
    // Find index of an element with an array
    int findIdx(vector<int> &A, const int &element){
        for(int i=0;i<(int)A.size();++i) if(A[i] == element) return i;
        return -1;
    }
    // Get a random number from 0 ~ 1
    LD getURand(){
        return ((LD)rand() / (RAND_MAX));
    }
    // Judge whether to swap or not
    bool needSwap(const LD& o, const LD& f){
        return f > res*LD(0.999);
    }
public:
    SantaGifts(char *wishfile, char *preffile){
        string info;
        ifstream fin;
        clock_t start_time, end_time;
        
        //Loading wish_list info
        start_time = clock();
        cout << "LOADING CHILDREN WISH LIST DATA!" <<endl;
        wish_score.resize(NUM_CHLD);
        fin.open(wishfile);
        for(int cidx=0;cidx<NUM_CHLD;++cidx){
            assert(getline(fin, info));
            auto j = info.find(',') + 1;
            int k = 0;
            while(j!=string::npos && k<WISH_SIZE){
                int gidx = stoi(info.substr(j));
                wish_score[cidx][gidx] = LD(WISH_SIZE-k)/WISH_SIZE;
                j = info.find(',', j) + 1;
                ++k;
            }
            assert(k == WISH_SIZE);
        }
        fin.close();
        end_time = clock();
        cout << "LOADING DATA TIME USAGE: "<<LD(end_time - start_time)/CLOCKS_PER_SEC << " s"<<endl;
        cout << "VERIFY WISH SCORES: (FIRST CHILD)" <<endl;
        cout << "SIZE = " << (int)wish_score[0].size() <<endl;
        for(auto p: wish_score[0]) cout<<'('<<p.first<<':'<<p.second<<") ";
        cout << endl << "VERIFICATION FINISHED" << endl << endl;
        
        //Loading pref_list info
        start_time = clock();
        cout << "LOADING SANTA PREFERENCE DATA!" <<endl;
        gift_score.resize(NUM_CHLD);
        fin.open(preffile);
        for(int gidx=0;gidx<NUM_GIFT;++gidx){
            assert(getline(fin, info));
            auto j = info.find(',') + 1;
            int k = 0;
            while(j!=string::npos && k<PREF_SIZE){
                int cidx = stoi(info.substr(j));
                assert(cidx < NUM_CHLD);
                gift_score[cidx][gidx] = LD(PREF_SIZE-k)/PREF_SIZE;
                j = info.find(',', j) + 1;
                ++k;
            }
            assert(k == PREF_SIZE);
        }
        fin.close();
        end_time = clock();
        cout << "LOADING DATA TIME USAGE: "<<LD(end_time - start_time)/CLOCKS_PER_SEC << " s"<<endl;
        cout << "VERIFY GIFT SCORES: (FIRST CHILD)" <<endl;
        cout << "SIZE = " << (int)gift_score[0].size() <<endl;
        for(auto p: gift_score[0]) cout<<'('<<p.first<<':'<<p.second<<") ";
        cout << endl << "VERIFICATION FINISHED" << endl << endl;
    }
    
    void init(){
        // Initialize picks
        clock_t start_time, end_time;
        cout<<"INITIAL ASSIGNMENT!"<<endl;
        
        start_time = clock();
        sum_ch = sum_sh = 0.;
        picks.resize(NUM_CHLD);
        unordered_map<int, int> cnt;
        unordered_set<int> unassigned;
        for(int i=0;i<NUM_GIFT;++i) cnt[i] = GIFT_LIMT;
        
        // Deal with triplets
        cout<<"ASSIGNING TRIPLETS!"<<endl;
        for(int childIdx=0; childIdx<triplet_lim; childIdx+=3){
            LD init_score = -100., wish_part = 0., gift_part = 0.;
            int gidx = -1;
            for(int delta = 0; delta < 3; ++delta){
                for(auto p: wish_score[childIdx+delta]) if(cnt.count(p.first) && cnt[p.first] >= 3){
                    LD tmp_wscore = 0.;
                    for(int q=0;q<3;++q) tmp_wscore += getChildWishScore(childIdx+q, p.first);
                    LD tmp_gscore = 0.;
                    for(int q=0;q<3;++q) tmp_gscore += getSantaGiftScore(childIdx+q, p.first);
                    LD tmp_score = powl(tmp_wscore, 3) + powl(tmp_gscore, 3);
                    if(tmp_score > init_score){
                        gidx = p.first;
                        init_score = tmp_score;
                        wish_part = tmp_wscore;
                        gift_part = tmp_gscore;
                    }
                }
            }
            for(int delta = 0; delta < 3; ++delta){
                for(auto p: gift_score[childIdx+delta]) if(cnt.count(p.first) && cnt[p.first] >= 3){
                    LD tmp_wscore = 0.;
                    for(int q=0;q<3;++q) tmp_wscore += getChildWishScore(childIdx+q, p.first);
                    LD tmp_gscore = 0.;
                    for(int q=0;q<3;++q) tmp_gscore += getSantaGiftScore(childIdx+q, p.first);
                    LD tmp_score = powl(tmp_wscore, 3) + powl(tmp_gscore, 3);
                    if(tmp_score > init_score){
                        gidx = p.first;
                        init_score = tmp_score;
                        wish_part = tmp_wscore;
                        gift_part = tmp_gscore;
                    }
                }
            }
            if(gidx < 0){
                for(auto p:cnt) if(p.second >= 3){
                    gidx = p.first;
                    break;
                }
                for(int q=0;q<3;++q) wish_part += getChildWishScore(childIdx+q, gidx);
                for(int q=0;q<3;++q) gift_part += getSantaGiftScore(childIdx+q, gidx);
            }
            picks[childIdx] = picks[childIdx+1] = picks[childIdx+2] = gidx;
            sum_ch += wish_part;
            sum_sh += gift_part;
            cnt[gidx] -= 3;
            if(!cnt[gidx]) cnt.erase(gidx);
        }
        
        // Dealing twins
        cout<<"ASSIGNING TWINS!"<<endl;
        for(int childIdx=triplet_lim; childIdx<twin_lim; childIdx+=2){
            LD init_score = -100., wish_part = 0., gift_part = 0.;
            int gidx = -1;
            for(int delta = 0; delta < 2; ++delta){
                for(auto p: wish_score[childIdx+delta]) if(cnt.count(p.first) && cnt[p.first] >= 2){
                    LD tmp_wscore = 0.;
                    for(int q=0;q<2;++q) tmp_wscore += getChildWishScore(childIdx+q, p.first);
                    LD tmp_gscore = 0.;
                    for(int q=0;q<2;++q) tmp_gscore += getSantaGiftScore(childIdx+q, p.first);
                    LD tmp_score = powl(tmp_wscore, 3) + powl(tmp_gscore, 3);
                    if(tmp_score > init_score){
                        gidx = p.first;
                        init_score = tmp_score;
                        wish_part = tmp_wscore;
                        gift_part = tmp_gscore;
                    }
                }
            }
            for(int delta = 0; delta < 2; ++delta){
                for(auto p: gift_score[childIdx+delta]) if(cnt.count(p.first) && cnt[p.first] >= 2){
                    LD tmp_wscore = 0.;
                    for(int q=0;q<2;++q) tmp_wscore += getChildWishScore(childIdx+q, p.first);
                    LD tmp_gscore = 0.;
                    for(int q=0;q<2;++q) tmp_gscore += getSantaGiftScore(childIdx+q, p.first);
                    LD tmp_score = powl(tmp_wscore, 3) + powl(tmp_gscore, 3);
                    if(tmp_score > init_score){
                        gidx = p.first;
                        init_score = tmp_score;
                        wish_part = tmp_wscore;
                        gift_part = tmp_gscore;
                    }
                }
            }
            if(gidx < 0){
                for(auto p:cnt) if(p.second >= 2){
                    gidx = p.first;
                    break;
                }
                for(int q=0;q<2;++q) wish_part += getChildWishScore(childIdx+q, gidx);
                for(int q=0;q<2;++q) gift_part += getSantaGiftScore(childIdx+q, gidx);
            }
            picks[childIdx] = picks[childIdx+1] = gidx;
            sum_ch += wish_part;
            sum_sh += gift_part;
            cnt[gidx] -= 2;
            if(!cnt[gidx]) cnt.erase(gidx);
        }
        
        // Deal with single children
        cout<<"ASSIGNING SINGLES!"<<endl;
        for(int childIdx=twin_lim; childIdx<NUM_CHLD; ++childIdx){
            LD init_score = -100., wish_part = 0., gift_part = 0.;
            int gidx = -1;
            for(auto p: wish_score[childIdx]) if(cnt.count(p.first)){
                LD tmp_wscore = getChildWishScore(childIdx, p.first);
                LD tmp_gscore = getSantaGiftScore(childIdx, p.first);
                LD tmp_score = powl(tmp_wscore, 3) + powl(tmp_gscore, 3);
                if(tmp_score > init_score){
                    gidx = p.first;
                    init_score = tmp_score;
                    wish_part = tmp_wscore;
                    gift_part = tmp_gscore;
                }
            }
            for(auto p: gift_score[childIdx]) if(cnt.count(p.first)){
                LD tmp_wscore = getChildWishScore(childIdx, p.first);
                LD tmp_gscore = getSantaGiftScore(childIdx, p.first);
                LD tmp_score = powl(tmp_wscore, 3) + powl(tmp_gscore, 3);
                if(tmp_score > init_score){
                    gidx = p.first;
                    init_score = tmp_score;
                    wish_part = tmp_wscore;
                    gift_part = tmp_gscore;
                }
            }
            if(gidx < 0){
                gidx = cnt.begin()->first;
                wish_part = getChildWishScore(childIdx, gidx);
                gift_part = getSantaGiftScore(childIdx, gidx);
            }
            picks[childIdx] = gidx;
            sum_ch += wish_part;
            sum_sh += gift_part;
            cnt[gidx] -= 1;
            if(!cnt[gidx]) cnt.erase(gidx);
        }
        res = current = powl(sum_ch/NUM_CHLD, 3) + powl(sum_sh/NUM_CHLD, 3);
        optimal = picks;
        assignment.resize(NUM_GIFT);
        for(int childIdx=0;childIdx<NUM_CHLD;++childIdx){
            assignment[optimal[childIdx]].push_back(childIdx);
        }
        cout<<"WISH_SCORE = "<<sum_ch/NUM_CHLD << ";  GIFT_SCORE = "<<sum_sh/NUM_CHLD<<endl;
        cout<<"INITAL_SCORE = "<< res <<endl;
        end_time = clock();
        cout << "INITIALIZATION TIME USAGE: "<<LD(end_time - start_time)/CLOCKS_PER_SEC << " s"<<endl;
        cout << "VERIFY ASSIGNMENT: ";
        for(auto vec: assignment) assert((int)vec.size() == GIFT_LIMT);
        cout << "SUCCESS!" <<endl;
    }
    
    void restart(char* fname){
        // Initialize picks
        string info;
        ifstream fin;
        clock_t start_time, end_time;
        cout<<"RESTARTING FROM AN EXISTING PROGRESS!"<<endl;
        cout<<"LOADING ASSIGNMENT FROM A CSV FILE!" <<endl;
        start_time = clock();
        sum_ch = sum_sh = 0.;
        picks.resize(NUM_CHLD);
        assignment.resize(NUM_GIFT);
        fin.open(fname);
        getline(fin, info);
        for(int cidx=0;cidx<NUM_CHLD;++cidx){
            assert(getline(fin, info));
            assert(cidx == stoi(info));
            auto j = info.find(',') + 1;
            int gidx = stoi(info.substr(j));
            assert(gidx < NUM_GIFT);
            picks[cidx] = gidx;
            assignment[gidx].push_back(cidx);
            sum_ch += getChildWishScore(cidx, gidx);
            sum_sh += getSantaGiftScore(cidx, gidx);
        }
        res = current = powl(sum_ch/NUM_CHLD, 3) + powl(sum_sh/NUM_CHLD, 3);
        optimal = picks;
        cout<<"WISH_SCORE = "<<sum_ch/NUM_CHLD << ";  GIFT_SCORE = "<<sum_sh/NUM_CHLD<<endl;
        cout<<"INITAL_SCORE = "<< res <<endl;
        end_time = clock();
        cout << "INITIALIZATION TIME USAGE: "<<LD(end_time - start_time)/CLOCKS_PER_SEC << " s"<<endl;
        cout << "VERIFY ASSIGNMENT: ";
        for(auto vec: assignment) assert((int)vec.size() == GIFT_LIMT);
        cout << "SUCCESS!" <<endl;
    }
    
    int oneStepEvolve(){
        int src_gidx = rand()%NUM_GIFT;
        int tar_gidx = rand()%NUM_GIFT;
        int k_pos = rand()%GIFT_LIMT;
        int k = assignment[src_gidx][k_pos];
        int n_swap = 1;
        vector<int> src_chld_pos, tar_chld_pos;
        if(k < triplet_lim){
            n_swap = 3;
            for(int q=0;q<3;++q){
                int tmp_idx = findIdx(assignment[src_gidx], (k/3)*3 + q);
                src_chld_pos.push_back(tmp_idx);
            }
        }
        else if(k < twin_lim){
            n_swap = 2;
            for(int q=1;q<=2;++q){
                int tmp_idx = findIdx(assignment[src_gidx], ((k-1)/2)*2 + q);
                src_chld_pos.push_back(tmp_idx);
            }
        }
        else src_chld_pos.push_back(k_pos);
        
        bool roll_success = false;
        while(!roll_success){
            while(tar_gidx == src_gidx) tar_gidx = rand()%NUM_GIFT;
            int roll_count = 0;
            set<int> record;
            while(roll_count < roll_lim && (int)tar_chld_pos.size() < n_swap){
                int tmp_idx = rand()%GIFT_LIMT;
                if(assignment[tar_gidx][tmp_idx] >= twin_lim && !record.count(tmp_idx)){
                    tar_chld_pos.push_back(tmp_idx);
                    record.insert(tmp_idx);
                }
                ++roll_count;
            }
            if(roll_count == roll_lim){
                tar_gidx = rand()%NUM_GIFT;
                tar_chld_pos.clear();
            }
            else break;
        }
        LD original_score = current;
        LD wish_diff = 0., gift_diff = 0.;
        for(int i=0;i<n_swap;++i){
            wish_diff -= getChildWishScore(assignment[tar_gidx][tar_chld_pos[i]], tar_gidx);
            wish_diff -= getChildWishScore(assignment[src_gidx][src_chld_pos[i]], src_gidx);
            wish_diff += getChildWishScore(assignment[tar_gidx][tar_chld_pos[i]], src_gidx);
            wish_diff += getChildWishScore(assignment[src_gidx][src_chld_pos[i]], tar_gidx);
            
            gift_diff -= getSantaGiftScore(assignment[tar_gidx][tar_chld_pos[i]], tar_gidx);
            gift_diff -= getSantaGiftScore(assignment[src_gidx][src_chld_pos[i]], src_gidx);
            gift_diff += getSantaGiftScore(assignment[tar_gidx][tar_chld_pos[i]], src_gidx);
            gift_diff += getSantaGiftScore(assignment[src_gidx][src_chld_pos[i]], tar_gidx);
        }
        LD final_score = powl((sum_ch+wish_diff)/NUM_CHLD, 3) + powl((sum_sh+gift_diff)/NUM_CHLD, 3);
        if(needSwap(original_score, final_score)){
            for(int i=0;i<n_swap;++i){
                int src_cidx = assignment[src_gidx][src_chld_pos[i]];
                int tar_cidx = assignment[tar_gidx][tar_chld_pos[i]];
                picks[src_cidx] = tar_gidx;
                picks[tar_cidx] = src_gidx;
                assignment[src_gidx][src_chld_pos[i]] = tar_cidx;
                assignment[tar_gidx][tar_chld_pos[i]] = src_cidx;
            }
            current = final_score;
            sum_ch += wish_diff;
            sum_sh += gift_diff;
            if(current > res){
                res = current;
                optimal = picks;
            }
            return 1;
        }
        return 0;
    }
    
    void evolution(const LD& epoch_time = 60.){
        clock_t start_time, end_time;
        cout<<"START OPTIMIZATION" <<endl;
        
        //Using time to divide epoches
        start_time = clock();
        int k = 1, swap_cnt = 0;
        while(LD(clock()-start_time)/CLOCKS_PER_SEC <= NUM_MINUTE * epoch_time){
            if(LD(clock()-start_time)/CLOCKS_PER_SEC > k*epoch_time){
                cout<<"=========================================="<<endl;
                cout<<"AT EPOCH "<< k << endl;
                cout<<"NUMBER OF SWAPS = " << swap_cnt << endl;
                cout<<"CURRENT SCORE = " << res << endl << endl;
                ++k;
                swap_cnt = 0;
            }
            swap_cnt += this->oneStepEvolve();
        }
        end_time = clock();
        cout << "OPTIMIZATION TIME USAGE: "<<LD(end_time - start_time)/CLOCKS_PER_SEC << " s"<<endl;
        return;
    }
    
    vector<int> getResult(){
        return this->optimal;
    }
    
    void judgeCorrect(){
        unordered_map<int, int> gift_count;
        cout<<"CHECKING CORRECTNESS: "<<endl;
        for(int i=0;i<triplet_lim;i+=3){
            assert(optimal[i] == optimal[i+1]);
            assert(optimal[i] == optimal[i+2]);
            gift_count[optimal[i]] += 3;
        }
        for(int i=triplet_lim;i<twin_lim;i+=2){
            assert(optimal[i] == optimal[i+1]);
            gift_count[optimal[i]] += 2;
        }
        for(int i=twin_lim;i<NUM_CHLD;++i){
            gift_count[optimal[i]]++;
        }
        for(auto p:gift_count) assert(p.second == GIFT_LIMT);
        cout << "THE INPUT/INITIALIZATION IS CORRECT!" <<endl;
    }
};

int main(int argc, char *argv[]){
    char wfname[] = "../input/santa-gift-matching/child_wishlist_v2.csv";
    char gfname[] = "../input/santa-gift-matching/gift_goodkids_v2.csv";
    auto *santa_gift = new SantaGifts(wfname, gfname);
    if(argc <= 1) santa_gift->init();
    else santa_gift->restart(argv[1]);
    
    // Check correctness
    santa_gift->judgeCorrect();
    santa_gift->evolution();
    
    // Check correctness
    santa_gift->judgeCorrect();
    auto ans = santa_gift->getResult();
    delete santa_gift;
    cout<<"WRITE INTO CSV FILE"<<endl;
    ofstream fout;
    fout.open ("output.csv");
    fout<<"ChildId,GiftId"<<endl;
    for(int i=0;i<NUM_CHLD;++i){
        fout<<i<<','<<ans[i];
        if(i<NUM_CHLD-1) fout<<endl;
    }
    fout.close();
    return 0;
}
""")
f.close()

os.system('g++ -std=c++11 -O3 test.cpp -o test')
os.system('./test ../input/pkugoodspeed/restart.csv')
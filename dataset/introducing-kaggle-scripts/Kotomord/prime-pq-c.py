src = '''
#include <queue>
#include <stdio.h>
#include <stdlib.h> 

using namespace std;

struct Pair{
   int point;
   int step;
   Pair(int a, int b){
       point = a;
       step = b;
   }
};

inline bool operator <(const Pair& p1, const Pair& p2){
   if(p1.point!=p2.point) return p1.point>p2.point;
   return p1.step<p2.step;
}


int main(){
    int root =  8424;
    int upbound = root*root;
    priority_queue<Pair>  queue;
    int cnt =1;
 FILE *f = fopen("primes.csv", "w");
    for(int i = 3; i<upbound; i+=2){
         if(queue.empty()){
            ++cnt;
          fprintf(f, "primes\\n2\\n%d\\n", i);
            queue.push(Pair(3*i, 2*i));
         }else if(queue.top().point == i){
           while(queue.top().point == i){
               Pair p = queue.top(); 
               queue.pop();
               p.point+=p.step;
               queue.push(p); 
           }
         }else{
          fprintf(f, "%d\\n", i); 
            if(i<root) queue.push(Pair(i*i, 2*i));
            ++cnt;
         }

    }
    printf("%d primes found\\n", cnt); 
}
'''

with open('go.cpp', 'w') as srcfile:
    srcfile.write(src)

from subprocess import call

call(["g++", "-O3", "go.cpp", "-o", "go"])
call(["./go"])



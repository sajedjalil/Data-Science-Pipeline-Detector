src = '''
#include <stdio.h>
#include <stdlib.h>

#define LIMIT  92949672
#define BUFFER_SIZE 30

char digitChars[10] = {'0','1','2','3','4','5','6','7','8','9'};

char* writeInt(unsigned long i, char *buffer) {
    char *cursor = buffer + BUFFER_SIZE -1;
    unsigned leastSigDig;
    
    *cursor = 0;
    do {
        leastSigDig = i % 10;
        i /= 10;
        cursor--;
        *cursor = digitChars[leastSigDig];
    } while(i != 0);
    return cursor;
}

int main() {
    unsigned long i,j,k;
    char *isPrime;
    char buf[BUFFER_SIZE];
    char *outputStr;
    
    
    isPrime = malloc(sizeof(char)*LIMIT);
    if(!isPrime){
        printf("Couldn't allocate memory :(\\n");
        return 1;
        }
        
    FILE *f = fopen("primes.csv", "w");
    fprintf(f, "Primes\\n2\\n");
    
    for (i=0; i<LIMIT; i++){
        isPrime[i]=1;
    }

    for (i=3; i < LIMIT*2; i += 2){
        if (isPrime[(i-3)/2]){
            for (j = i*i; j < LIMIT*2; j += 2*i)
                isPrime[(j-3)/2] = 0;
        }
    }
    
    unsigned int n_primes = 0;
    for (i=3; i < LIMIT*2; i += 2)
        if (isPrime[(i-3)/2]) {
            n_primes++;
            outputStr = writeInt(i, buf);
            fprintf(f, "%s\\n", outputStr);
        }
    fclose(f);
    printf("Found %d primes.\\n", n_primes);
return 0;
}
'''

with open('go.c', 'w') as srcfile:
    srcfile.write(src)

from subprocess import call

call(["gcc", "-O3", "go.c", "-o", "go"])
call(["./go"])



def gcd(a,b):
    if(b==0):
        return a
    return gcd(b,a%b)

if __name__ == '__main__':
    val = gcd(10,3)
    
    
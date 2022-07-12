"""
sum: 34894.26651   zeros: 56


"""


coal=0
book=0
bike=0
train=0
blocks=0
doll=0
horse=0
gloves=0
ball=0
scenario={}
scenario["coal_book"]=0
scenario["ball_doll"]=0
scenario["gloves_doll"]=0
specialc=0
with open("Santa_05.csv", 'w') as f:
        f.write("Gifts\n")
        for i in range(1000):
                
             # mul("doll",3)+mul("horse",3)+mul("ball",5)+mul("book",1) 37
            if book < 1000 and  doll < 997 and ball < 1080:
                f.write('doll_'+str(doll))
                doll+=1
                f.write(' doll_'+str(doll))
                doll+=1
                f.write(' doll_'+str(doll))
                doll+=1
                
                f.write(' horse_'+str(horse))
                horse+=1
                f.write(' horse_'+str(horse))
                horse+=1
                f.write(' horse_'+str(horse))
                horse+=1
                
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' book_'+str(book)+'\n')
                book+=1
            #mul("train",1)+mul("blocks",1)+mul("horse",1)+mul("doll",1)+mul("gloves",4) 35
            elif book < 1000 and bike < 500 and gloves < 197:
                f.write('train_'+str(train))
                train+=1
                f.write(' blocks_'+str(blocks))
                blocks+=1
                f.write(' horse_'+str(horse))
                horse+=1
                f.write(' doll_'+str(doll))
                doll+=1
                for j in range(0,4):
                    f.write(' gloves_'+str(gloves))
                    gloves+=1
                f.write("\n")
            
            
            # mul("train",1)+mul("train",1)+mul("blocks",1)+mul("doll",1) 34
            elif blocks < 1000 and bike < 500 and doll < 1000: 
                f.write('train_'+str(train))
                train+=1
                f.write(' train_'+str(train))
                train+=1
                f.write(' blocks_'+str(blocks))
                blocks+=1
                f.write(' doll_'+str(doll))
                doll+=1
                f.write("\n")
            
            
                
            #  mul("blocks",1)+mul("bike",1)+mul("horse",1) #31    
            elif blocks < 1000 and bike < 500 and book < 960 and horse < 894: #894
                f.write('blocks_'+str(blocks))
                blocks+=1
                f.write(' bike_'+str(bike))
                bike+=1
                f.write(' horse_'+str(horse))
                horse+=1
                f.write("\n")
            # mul("book",20)  32
            elif  specialc < 11:
                specialc+=1
                f.write('book_'+str(book))
                book+=1
                for j in range(0,19):
                    f.write(' book_'+str(book))
                    book+=1
                f.write("\n")
                
            # mul("blocks",2)+mul("train",1)+mul("book",2) 35
            elif blocks < 1000  and book < 1000 and train < 1000: 
                f.write('blocks_'+str(blocks))
                blocks+=1
                f.write(' blocks_'+str(blocks))
                blocks+=1
                f.write(' train_'+str(train))
                train+=1
                f.write(' book_'+str(book))
                book+=1
                f.write(' book_'+str(book))
                book+=1
                f.write("\n")
                
            
            
            
                
            
                

print("coal_book (820)",scenario["coal_book"])
print("ball_doll(1684): ",scenario["ball_doll"])
print("glove_doll  :",scenario["gloves_doll"])
print("\n\n")

print("coal max(166)",coal)                
print("horse max(1000)",horse)
print("book max(1200)",book)
print("bike max(500)",bike)
print("gloves max(200)",gloves)
print("train max(1000)",train)
print("ball max(1100)",ball)
print("doll max(1000)",doll)
print("blocks max(1000)",blocks)



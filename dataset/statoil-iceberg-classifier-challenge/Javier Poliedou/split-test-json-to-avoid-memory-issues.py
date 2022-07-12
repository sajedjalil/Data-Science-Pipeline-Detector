## SPLITJSON select "nbimg" images in the JSON called "namein" and dump it
## in the JSON called "nameout". It starts from caracter number "num_car_dep" 
## and records whenever the first open brace ("{") appear (which means, a new image)

def SplitJSON(namein,nameout,num_car_dep,nbimg):
  f = open(namein,'r')
  fres = open(nameout,'w')
  f.read(num_car_dep)
  nb_brace_in=0
  nb_brace_out=0
  nb_car=num_car_dep
  res=""
  car=f.read(1)
  while ((nb_brace_in<=nbimg) and (car!="")):
    nb_car+=1
    if (car=="{"):
      nb_brace_in+=1
    if (car=="}"):
      nb_brace_out+=1
    res+=car
    car=f.read(1)
  fres.write("["+res[:-2]+"]")
  fres.close()
  f.close()
  return nb_car-1

## SplitOneJSON splits a JSON file containing xxl images into xx JSON files containing 1 image each
## to avoid RAM issues using PANDA on the "big" file "test.json" with the command : pd.read_json('test.json').

def SplitOneJSON(namein,nameoutgen,num_car_dep_init,numax):
 numiter=0 
 num_car_dep=-1
 while((numiter<=numax) and (num_car_dep!=num_car_dep_init)):
   print("Iteration num = "+str(numiter)+"\n")
   num_car_dep=num_car_dep_init
   num_car_dep_init=SplitJSON(namein,nameoutgen+str(numiter)+'.json',num_car_dep,1)
   numiter+=1

## How to use it : namein="trainN3.json" (JSON with the 3 first images of train.json), nameoutgen="train-split_" (generic name of the different files, each containing 1 image only)
## num_car_dep_init=1 (start after the first caracter which is a "["), numax=9000 (number of image max. If it is > than the real max number
## of images, it will stop when it reaches the maximum number of images)

#SplitOneJSON("trainN3.json","train-split_",1,10)



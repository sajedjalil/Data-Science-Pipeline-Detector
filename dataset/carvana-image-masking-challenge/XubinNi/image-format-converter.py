from PIL import Image
import sys, getopt
import os

def main(argv):
    inputfile = ''
    outputfile = ''
    format_from=''
    format_to=''
    try:
        opts, args = getopt.getopt(argv,"hi:o:f:t:",["ifile=","ofile=","from=","to="])
    except getopt.GetoptError:
        print ('test.py -i <inputfile> -o <outputfile> -f<format_from> -t<format_to>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-f", "--from"):
            format_from = arg
        elif opt in ("-t", "--to"):
            format_to = arg
        

    if(inputfile=='' or outputfile=='' or format_from=='' or format_to==''):
        print ('test.py -i <inputfile> -o <outputfile> -f<format_from> -t<format_to>')
        sys.exit(2)
    else:
        print('Input folder:'+inputfile)
        print('Output folder:'+outputfile)
        print('From:'+format_from)
        print('To:'+format_to)


    file_list=[]
    for file in os.listdir(inputfile):
            if file.endswith(format_from):
                    file_list.append(os.path.join(inputfile, file))


    for file in file_list:
        outfile_name=file.split('.', 1 )[0][:-5]+'.'+format_to
        #outfile_name=file.split('.', 1 )[0]+'.'+format_to
        outfile_name=outfile_name.replace(inputfile,outputfile)

        print("Formating "+file+" to "+outfile_name)
        Image.open(file).convert('RGB').save(outfile_name)




if __name__ == "__main__":
   main(sys.argv[1:])
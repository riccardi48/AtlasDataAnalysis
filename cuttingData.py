#The decoded data is sometimes repeated including the header, the header starts with a hashtag. This finds the repeats and cuts them.
#Warning: Large decoded data files take a lot of memory. 5Gb + might won't work.

from glob import glob

pathToOrigin = "/home/atlas/rballard/for_magda/data/202204*udp*_decode.dat"
pathToOutput = "/home/atlas/rballard/for_magda/data/Cut/202204*udp*_decode.dat"
files = glob(pathToOrigin)
files_done = glob(pathToOutput)
for file in files:
    to_be_saved_as =  "/".join(file.split("/")[:-1])+"/Cut/"+file.split("/")[-1].split(".")[0]+".dat"
    if to_be_saved_as not in files_done:
        with open(file, 'rb') as inp, open(to_be_saved_as, 'w') as out:
            print(f"Opening {file}")
            string = "#" + str(inp.read()).split("#")[1].replace('\\n', '\n').replace('\\t', '\t')
            if string[-1] == "'":
                string = string[:-1]
            out.write(string)
            print(f"Saved {to_be_saved_as}")
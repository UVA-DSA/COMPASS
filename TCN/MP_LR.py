
import os, sys
global unique_labels
unique_labels = {}

global invalid_States
invalid_States = {}



from itertools import accumulate

mathematicaColors = {
    "blue":"#5E81B5",
    "orange":"#E09C24",
    "red":"#EA5536",
    "purple":"#A5609D",
    "green":"#8FB131",
    "blue" :"#5e9ec8",
    "olive":"#929600",
    "terracotta":"#C56E1A",
    "yellow":"#FEC000",
}
# list of colors for the annotations
colors =["#5E81B5","#D47BC9","#7CEB8E","#E36D6D","#C9602A","#77B9E0","#A278F0","#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5","#D47BC9","#7CEB8E","#E36D6D","#C9602A","#77B9E0","#A278F0","#D66F6D","#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5"]

def main():
    dir=os.getcwd()
    task = ""
    try:
        task=sys.argv[1]
        #print(task)
    except:
        print("Error: no task provided \nUsage: python MP_LR.py <task>")
        available_tasks = next(os.walk(dir))[1]
        print("Available task transcripts: ", available_tasks)
        sys.exit()

    I = Iterator(task)
    #I.SeparateMP();
    I.SeparateMP();

    quit();

class Iterator:
    '''
    __init__(self,task)

    uses task String argument {Knot_Tying, Suturing, Needle_Passing}
    to set the folder path containing combined transcripts
    and locations to store the L and R transcripts


    '''
    def __init__(self, task):
        self.CWD = os.path.dirname(os.path.realpath(__file__))
        self.task = task

        self.mpdir = os.path.join(self.CWD, task,"motion_primitives_combined_K")
        self.mpdir_R = os.path.join(self.CWD, task,"motion_primitives_RE")
        self.mpdir_L = os.path.join(self.CWD, task,"motion_primitives_LE")

    def SeparateMP(self):
        count = 0
        for root, dirs, files in os.walk(self.mpdir):
            for file in files:
                mpfile = os.path.join(self.mpdir, file)
                mpfile_R = os.path.join(self.mpdir_R, file)
                mpfile_L = os.path.join(self.mpdir_L, file)
                if os.path.exists(mpfile_R):
                    os.remove(mpfile_R)
                if os.path.exists(mpfile_L):
                    os.remove(mpfile_L)
                mplines = []
                mplines_R = []
                mplines_L = []
                with open(mpfile) as f:
                    for line in f:
                        mplines.append(line.rstrip())
                mplines_R,mplines_L = self.separateLines(mplines,file);
                self.save(mpfile_L,mplines_L)
                self.save(mpfile_R,mplines_R)

    def separateLines(self, lines,fname):
        L_lines = []
        R_lines = []

        R_count = 0
        L_count = 0
        for i in range(len(lines)):
            if("(R," in lines[i]):
                R_count+=1
            elif("(L," in lines[i]):
                L_count +=1
        R_lines.append(lines[0])
        L_lines.append(lines[0])
        lastGrasp = ""
        for i in range(1,len(lines)):
            l = lines[i]
            l_s = l.split(" ")
            #if("Grasp"in l and "Ring" in l):
            #    print("WORRY",fname)
                # we are grasping the ring,
            if("(R," in l and "(L," in l):
                # two MPs in one line, given R and L, split is simple
                if(l.find("(L,") < l.find("(R,")):
                    l_l = [l_s[0],l_s[1],l_s[2],l_s[3]]
                    l_r = [l_s[0],l_s[1],l_s[4],l_s[5]]
                    R_lines.append( " ".join(l_r))
                    L_lines.append( " ".join(l_l))
                else:
                    l_l = [l_s[0],l_s[1],l_s[4],l_s[5]]
                    l_r = [l_s[0],l_s[1],l_s[2],l_s[3]]
                    R_lines.append( " ".join(l_r))
                    L_lines.append( " ".join(l_l))
                    print("R and L but swapped idx",i+1,fname)
                if("Grasp(R, Needle)" in l or "Grasp(R, Ball/Block/Sleeve)" in l):
                    lastGrasp = "R"
                elif("Release(R, Needle)" in l or "Release(R, Ball/Block/Sleeve)" in l):
                    lastGrasp = "L"
                if("Grasp(L, Needle)" in l or "Grasp(L, Ball/Block/Sleeve)" in l):
                    lastGrasp = "L"
                elif("Release(L, Needle)" in l or "Release(L, Ball/Block/Sleeve)" in l):
                    lastGrasp = "R"

            elif("(LR" in l):
                l_l = l_s
                l_r = l_s
                R_lines.append( " ".join(l_r))
                L_lines.append( " ".join(l_l))
                lastGrasp = "L"
            elif("(RL" in l):
                l_l = l_s
                l_r = l_s
                R_lines.append( " ".join(l_r))
                L_lines.append( " ".join(l_l))
                lastGrasp = "R"
            elif(l.count("(R,") == 2):
                # two Rs
                l_l = [l_s[0],l_s[1],"Idle(L)"]
                l_r = l_s
                R_lines.append( " ".join(l_r))
                L_lines.append( " ".join(l_l))
                if("Grasp(R, Needle)" in l or "Grasp(R, Ball/Block/Sleeve)" in l):
                    lastGrasp = "R"
                elif("Release(R, Needle)" in l or "Release(R, Ball/Block/Sleeve)" in l):
                    lastGrasp = "L"
            elif(l.count("(L,") == 2):
                l_l = l_s
                l_r = [l_s[0],l_s[1],"Idle(R)"]
                R_lines.append( " ".join(l_r))
                L_lines.append( " ".join(l_l))
                if("Grasp(L, Needle)" in l or "Grasp(L, Ball/Block/Sleeve)" in l):
                    lastGrasp = "L"
                elif("Release(L, Needle)" in l or "Release(L, Ball/Block/Sleeve)" in l):
                    lastGrasp = "R"

            elif(l.count("(") == 2):
                # there are two actions, one is one is L or R and the other is another action
                if(l.find("(L,") != -1):
                    # L is present, the other action is dependent on lastGrasp
                    if(lastGrasp == ""):
                        print("Neither Left or Right - an no antecedent Index",i,"in file")
                    elif(lastGrasp == "R"):
                        l_l = [l_s[0],l_s[1],l_s[2],l_s[3]]
                        l_r = [l_s[0],l_s[1],l_s[4],l_s[5]]
                        R_lines.append( " ".join(l_r))
                        L_lines.append( " ".join(l_l))
                    elif(lastGrasp == "L"):
                        l_l = l_s
                        l_r = [l_s[0],l_s[1],"Idle(R)"]
                        R_lines.append( " ".join(l_r))
                        L_lines.append( " ".join(l_l))
                elif( l.find("(R,") != -1):
                    # R is present, the other action is dependent on lastGrasp
                    if(lastGrasp == ""):
                        print("Neither Left or Right - an no antecedent Index",i,"in file")
                    elif(lastGrasp == "R"):
                        l_l = [l_s[0],l_s[1],"Idle(L)"]
                        l_r = l_s
                        R_lines.append( " ".join(l_r))
                        L_lines.append( " ".join(l_l))
                    elif(lastGrasp == "L"):
                        l_r = [l_s[0],l_s[1],l_s[2],l_s[3]]
                        l_l = [l_s[0],l_s[1],l_s[4],l_s[5]]
                        R_lines.append( " ".join(l_r))
                        L_lines.append( " ".join(l_l))
                        print("ever happens that there are two ( ( and R is present and last grab is L?",i+1,fname)
                else:
                    print("Two ( ( but neither L or R in",i,fname)
                if("Grasp(R, Needle)" in l or "Grasp(R, Ball/Block/Sleeve)" in l):
                    lastGrasp = "R"
                elif("Release(R, Needle)" in l or "Release(R, Ball/Block/Sleeve)" in l):
                    lastGrasp = "L"
                if("Grasp(L, Needle)" in l or "Grasp(L, Ball/Block/Sleeve)" in l):
                    lastGrasp = "L"
                elif("Release(L, Needle)" in l or "Release(L, Ball/Block/Sleeve)" in l):
                    lastGrasp = "R"

            # not two ( (
            elif("(L," in l):
                l_l = l_s
                l_r = [l_s[0],l_s[1],"Idle(R)"]
                R_lines.append( " ".join(l_r))
                L_lines.append( " ".join(l_l))
                if("Grasp(L, Needle)" in l or "Grasp(L, Ball/Block/Sleeve)" in l):
                    lastGrasp = "L"
                elif("Release(L, Needle)" in l or "Release(L, Ball/Block/Sleeve)" in l):
                    lastGrasp = "R"
            elif("(R," in l):
                l_l = [l_s[0],l_s[1],"Idle(L)"]
                l_r = l_s
                R_lines.append( " ".join(l_r))
                L_lines.append( " ".join(l_l))
                if("Grasp(R, Needle)" in l or "Grasp(R, Ball/Block/Sleeve)" in l):
                    lastGrasp = "R"
                elif("Release(R, Needle)" in l or "Release(R, Ball/Block/Sleeve)" in l):
                    lastGrasp = "L"



            else:
                #print("neither of every other case",i,fname)
                if(lastGrasp == ""):
                    print("Neither Left or Right - an no antecedent Index",i,"in file",fname)
                elif(lastGrasp == "R"):
                    l_l = [l_s[0],l_s[1],"Idle(L)"]
                    l_r = l_s
                    R_lines.append( " ".join(l_r))
                    L_lines.append( " ".join(l_l))

                elif(lastGrasp == "L"):
                    l_l = l_s
                    l_r = [l_s[0],l_s[1],"Idle(R)"]
                    R_lines.append( " ".join(l_r))
                    L_lines.append( " ".join(l_l))


        l_in_r = 0;
        r_in_l = 0;
        for t in R_lines:
            if("(L," in t):
                l_in_r+=1
        for t in L_lines:
            if("(R," in t):
                r_in_l+=1
        print(self.task,"CHECKSUM",len(lines) - (1+R_count+L_count), len(lines) - len(R_lines), len(lines) - len(L_lines),l_in_r,r_in_l)

        return R_lines,L_lines

    def save(self, x_file, x_lines):
        with open(x_file, 'w+') as f:
            for item in x_lines:
                f.write("%s\n" % item)

main();

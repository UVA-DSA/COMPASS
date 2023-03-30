# Kay Hutchinson 6/28/2021
#
# JIGSAWS_New_Gestures
#
# For relabing the videos in the JIGSAWS dataset using my new gesture
# definitions as described in the ReadMe.docx.
#
# Set the dataset folder location and choose which gesture to segment
# Location of dataset folder
#dir = "/home/student/Documents/Research/MICCAI_2022"
#
# Task - pick which task to segment
#task = "Suturing"
#task = "Needle_Passing"
#task = "Knot_Tying"
#task = "Pea_on_a_Peg"
#
#

import os
import sys
import glob
import cv2
from tkinter import *
import PIL
from PIL import Image
from PIL import ImageTk
import pathlib

global frameNum
global startFrame
global run
global VIDEO_LENGTH
#global t

# Get task from command line

#task = "Knot_Tying"

# Location of dataset folder
#dir=os.getcwd()
#dir = os.path.join(dir, "MPS")
dir = os.path.dirname(os.path.realpath(__file__))  

# List of gestures
#actions = ["Idle", "Hold", "Grasp", "Release", "PickUp", "PutDown", "Pull", "Push", "Exchange", "Exchange", "Unknown"]

actions = ["Touch","Grasp","Release","Untouch","Push","Pull"]

objects = ["Nothing", "Ball/Block/Sleeve", "Needle", "Thread", "Fabric/Tissue", "Ring", "Post","Other"]

params1 = ["L","R","Peg","2","3"]


# List of objects
#objects=["Needle", "Thread", "Tissue", "Block", "Ball", "Ring", "Other", "None"]

# App code
# From https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Set up layout of widgets
        self.toptop = Frame(window)
        self.top = Frame(window)
        self.middle = Frame(window)
        self.middle2 = Frame(window)
        self.middle3 = Frame(window)
        self.bottom = Frame(window)
        self.bottom2 = Frame(window)
        self.bottom3 = Frame(window)
        self.toptop.pack(side=TOP,expand=True)
        self.top.pack(expand=True)
        self.middle.pack(expand=True)
        self.middle2.pack(expand=True)
        self.middle3.pack(expand=True)
        self.bottom.pack(expand=True)
        self.bottom2.pack(expand=True)
        self.bottom3.pack(side=BOTTOM, expand=True)

        # Note to not use x button to close window
        self.warning = Text(window, width=37, height=1)
        self.warning.insert(END,"DO NOT close window, use QUIT button")
        self.warning.pack(in_=self.toptop, side=RIGHT, anchor=E)

        # Create a canvas that can fit the above video source size
        w = int((480.0/self.vid.height)*self.vid.width)
        self.canvas = Canvas(window, width = w, height = 480)  #self.vid.width, height = self.vid.height)
        self.canvas.pack(in_=self.top, expand=True)

        # Move through frames with slider and buttons
        self.slider=Scale(window,from_=0,to=self.vid.length,orient=HORIZONTAL,length=600,command=self.slide,label="Frame:")
        self.slider.set(0)
        self.slider.pack(in_=self.middle, anchor=CENTER)

        self.next10_button=Button(window,text="Next 10", width=15, font='sans 15',command=self.next10)
        self.next10_button.pack(in_=self.middle, side=RIGHT, anchor=E)
        self.next_button=Button(window,text="Next", width=15, font='sans 15',command=self.next)
        self.next_button.pack(in_=self.middle, side=RIGHT, anchor=E)
        self.back10_button=Button(window,text="Back 10", width=15, font='sans 15',command=self.back10)
        self.back10_button.pack(in_=self.middle, side=LEFT, anchor=W)
        self.back_button=Button(window,text="Back", width=15, font='sans 15',command=self.back)
        self.back_button.pack(in_=self.middle, side=LEFT, anchor=W)

        # Radio button for gesture selection
        self.g = IntVar()
        #actions = ["Touch","Grasp","Release","Untouch","Push","Pull"]
        gestures=[("Touch",1),("Grasp",2),("Release",3),("Untouch",4),("Push",5),("Pull",6)]
        #gestures=[("Idle",1), ("Hold",2), ("Grasp",3), ("Release",4), ("PickUp",5), ("PutDown",6), ("Pull",7), ("Push",8), ("Exchange L->R",9), ("Exchange R->L",10), ("Unknown",11)]
        for gesture, val in gestures:
            self.gestureselect=Radiobutton(window,text=gesture,indicatoron=0,width=15,padx=5,pady=5,variable=self.g,value=val)
            self.gestureselect.pack(in_=self.middle2, side=LEFT, anchor=W,pady=5)

        # Radio button for object selection
        self.o = IntVar()
        #objects = ["Nothing", "Ball/Block/Sleeve", "Needle", "Thread", "Fabric/Tissue", "Ring", "Other"]

        objects=[("Nothing",1),("Ball/Block/Sleeve",2),("Needle",3),("Thread",4),("Fabric/Tissue",5),("Ring",6),("Post",7),("Other",8)]
        #objects=[("Needle",1), ("Thread",2), ("Tissue",3), ("Block",4), ("Ball",5), ("Ring",6), ("Other",7), ("None",8)]
        for object, val in objects:
            self.objectselect=Radiobutton(window,text=object,indicatoron=0,width=10,padx=10,pady=5,variable=self.o,value=val)
            self.objectselect.pack(in_=self.bottom, side=LEFT, anchor=W)

        # Radio button for gesture selection
        self.param1 = IntVar()
        #params1 = ["L","R","1","2","3"]
        params1 = [("L",1),("R",2),("Peg",3),("2",4),("3",5)]
        #gestures=[("Idle",1), ("Hold",2), ("Grasp",3), ("Release",4), ("PickUp",5), ("PutDown",6), ("Pull",7), ("Push",8), ("Exchange L->R",9), ("Exchange R->L",10), ("Unknown",11)]
        for parm, val in params1 :
            self.parmSelect=Radiobutton(window,text=parm,indicatoron=0,width=15,padx=5,pady=5,variable=self.param1,value=val)
            self.parmSelect.pack(in_=self.middle3, side=LEFT, anchor=W,pady=5)

        


        # Enter button to confirm choice
        self.enter = Button(window,text="Save MP and continue",width=30, font='sans 15',command=self.mark)
        self.enter.pack(in_=self.bottom2,anchor=CENTER)

        # Done button to finish video
        self.done=Button(window,text="Next video", width=15, font='sans 15',command=self.done)
        self.done.pack(in_=self.bottom2, anchor=CENTER)

        # Quit button to close properly
        self.quit=Button(window,text="Quit", font='sans 15',command=self.quit)
        self.quit.pack(in_=self.bottom3, anchor=CENTER)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        window.bind("<a>",self.back)
        window.bind("<d>",self.next)

        window.bind("<q>",self.back10)
        window.bind("<e>",self.next10)

        self.window.mainloop()

    def slide(self,f):
        global frameNum
        frameNum = min(int(f), VIDEO_LENGTH-1)

    def next(self, etc=False):
        global frameNum
        frameNum = min(frameNum + 1,VIDEO_LENGTH-1)
        self.slider.set(frameNum)

    def back(self, etc=False):
        global frameNum
        frameNum = max(frameNum - 1,0)
        self.slider.set(frameNum)

    def next10(self, etc=False):
        global frameNum
        frameNum = min(frameNum + 10,VIDEO_LENGTH-1)
        self.slider.set(frameNum)

    def back10(self, etc=False):
        global frameNum
        frameNum = max(frameNum - 10,0)
        self.slider.set(frameNum)

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        nextImg = PIL.Image.fromarray(frame)

        # Scale new image to height of 480 px, keep ratio
        width, height = nextImg.size
        scaling = 480.0/height  # scale to height of 480 px
        newwidth = int(scaling*width)
        nextImg=nextImg.resize((newwidth, 480))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = nextImg)
            self.canvas.create_image(0, 0, image = self.photo, anchor = NW)

        self.window.after(self.delay, self.update)

    def mark(self):
        global frameNum
        global startFrame
        #global t
        print("MARKING",t.name)

        # Write to transcript .txt file
        startF = str(int(startFrame))
        frameN = str(int(frameNum))
        action = str(actions[self.g.get()-1])
        p1 = str(params1[self.param1.get()-1])
        p2 = str(objects[self.o.get()-1])

        t.write(startF + " " + frameN + " " + action + "(" + p1+", " + p2 +")\n");

        '''
        if (str(actions[self.g.get()-1]))=="Idle":
            t.write(str(startFrame) + " " + str(frameNum) + " " + str(actions[self.g.get()-1]) + "(" + tool + ")\n")
        elif self.g.get()==9:
            t.write(str(startFrame) + " " + str(frameNum) + " " + str(actions[self.g.get()-1]) + "(L, R, " + str(objects[self.o.get()-1]) + ")\n")
        elif self.g.get()==10:
            t.write(str(startFrame) + " " + str(frameNum) + " " + str(actions[self.g.get()-1]) + "(R, L, " + str(objects[self.o.get()-1]) + ")\n")
        else:
            t.write(str(startFrame) + " " + str(frameNum) + " " + str(actions[self.g.get()-1]) + "(" + tool + ", " + str(objects[self.o.get()-1]) + ")\n")
        '''

        '''
        # Save image
        ret, frame = self.vid.get_frame()
        if ret:
            img1 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            imgpil = ImageTk.getimage( img1 )
            rgb_im = imgpil.convert('RGB')
            rgb_im.save(os.path.join(dir,"Transitions", str(frameNum)+".jpg"), "JPEG" )
        '''

        frameNum = min(frameNum + 1,VIDEO_LENGTH-1)
        startFrame = frameNum
        self.slider.set(frameNum)

    def done(self):
        t.close()
        self.window.destroy()

    def quit(self):
        global run
        # Delete unfinished transcript
        t.close()
        os.remove(t.name)
        self.window.destroy()
        run = 0

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # Get video length in frames
        global VIDEO_LENGTH
        VIDEO_LENGTH = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.length = VIDEO_LENGTH

    def get_frame(self):
        if self.vid.isOpened():
            self.vid.set(1, frameNum)
            ret, frame = self.vid.read()
            if ret:
                '''
                # Scale new image to height of 480 px, keep ratio
                nextImg=frame
                scaling = 480/self.height  # scale to height of 480 px
                newwidth = int(scaling*self.width)
                #nextImg.resize((newwidth, 480))
                frame=cv2.resize(frame, (newwidth, 480), interpolation=cv2.INTER_AREA)
                '''
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

task="Peg_Transfer" # default
try:
    task=sys.argv[1]
    print(task)
except:
    print("Error: invalid task\nUsage: python gesture_segmentation_labeling.py <task>\nTasks: Suturing, Needle_Passing, Knot_Tying, Pea_on_a_Peg, Post_and_Sleeve, Wire_Chaser_I, Peg_Transfer")


# Transcript and video directories
dir=os.path.dirname(os.getcwd())

taskDir = os.path.join(dir, "Datasets", "dV", task)
transcriptDir = os.path.join(taskDir,"transcriptions_mp")
if(not os.path.isdir(transcriptDir)):
    path = pathlib.Path(transcriptDir)
    path.mkdir(parents=True, exist_ok=True)
videoDir = os.path.join(taskDir,"video")

# List of finished transcripts
doneList = [done.split('\\')[-1].rsplit(".txt")[0] for done in glob.glob(transcriptDir+'\\*.txt')]
#print(doneList)

# Set to run
run = 1

# For each video ending in "_capture1.avi" or ".mp4"
#print(videoDir)
videos = glob.glob(videoDir+"/*.avi") + glob.glob(videoDir+"/*.mp4")
#print("len",len(videos ))
for video in videos:

    for tool in ["L", "R"]:
        #print("hello2")
        #print(tool)
        trial = video.split("\\")[-1]
        # Get name for transcript file name
        #trialName = trial.rsplit("_",1)[0]+"_"+tool
        trialName = trial.rsplit("_",1)[0]

        if task in ["Pea_on_a_Peg", "Post_and_Sleeve", "Wire_Chaser_I"]:   # single camera videos need different rsplit indexing
            #trialName = trial.rsplit(".")[0]
            #trialName = trial.rsplit(".")[0]+"_"+tool
            trialName = trial.rsplit(".")[0]

        if trialName in doneList:
            continue

        transcriptName = trialName+".txt"
        transcriptPath = os.path.join(transcriptDir, transcriptName)
        print("Working on: "+trialName)

        # Init frameNum counter
        frameNum = 0
        startFrame = 0

        # Create transcript file for this video
        
        t = open(transcriptPath, 'w+')
        #t.write("0 ")

        # Run app
        App(Tk(), trialName, video)

        # Closes all the frames
        t.close()
        doneList.append(trialName)
        cv2.destroyAllWindows()

        if run == 0:
            break
    if run == 0:
        break

        #print(transcriptPath)

# end
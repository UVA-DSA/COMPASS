# Kay Hutchinson 2/24/2021
#
# JIGSAWS gesture labeling
#
# For labing the videos in the JIGSAWS dataset using their gesture
# definitions.
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
from Tkinter import *
import PIL
from PIL import Image
from PIL import ImageTk

global frameNum
global startFrame
global run


# List of gestures
#actions = ["Idle", "Hold", "Grasp", "Release", "PickUp", "PutDown", "Pull", "Push", "Exchange", "Exchange", "Unknown"]
actions = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G13", "G14", "G15"]

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
        self.bottom = Frame(window)
        self.bottom2 = Frame(window)
        self.bottom3 = Frame(window)
        self.toptop.pack(side=TOP,expand=True)
        self.top.pack(expand=True)
        self.middle.pack(expand=True)
        self.middle2.pack(expand=True)
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
        #gestures=[("Idle",1), ("Hold",2), ("Grasp",3), ("Release",4), ("PickUp",5), ("PutDown",6), ("Pull",7), ("Push",8), ("Exchange L->R",9), ("Exchange R->L",10), ("Unknown",11)]
        gestures = [("G1",1), ("G2",2), ("G3",3), ("G4",4), ("G5",5), ("G6",6), ("G7",7), ("G8",8), ("G9",9), ("G10",10), ("G11",11), ("G12",12), ("G13",13), ("G14",14), ("G15",15)]
        for gesture, val in gestures:
            self.gestureselect=Radiobutton(window,text=gesture,indicatoron=0,width=5,padx=5,pady=5,variable=self.g,value=val)
            self.gestureselect.pack(in_=self.middle2, side=LEFT, anchor=W)

        # # Radio button for object selection
        # self.o = IntVar()
        # objects=[("Needle",1), ("Thread",2), ("Tissue",3), ("Block",4), ("Ball",5), ("Ring",6), ("Other",7), ("None",8)]
        # for object, val in objects:
        #     self.objectselect=Radiobutton(window,text=object,indicatoron=0,width=10,padx=10,pady=5,variable=self.o,value=val)
        #     self.objectselect.pack(in_=self.bottom, side=LEFT, anchor=W)

        # Enter button to confirm choice
        self.enter = Button(window,text="Save gesture and continue",width=30, font='sans 15',command=self.mark)
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

        self.window.mainloop()

    def slide(self,f):
        global frameNum
        frameNum = int(f)

    def next(self):
        global frameNum
        frameNum = frameNum + 1
        self.slider.set(frameNum)

    def back(self):
        global frameNum
        frameNum = frameNum - 1
        self.slider.set(frameNum)

    def next10(self):
        global frameNum
        frameNum = frameNum + 10
        self.slider.set(frameNum)

    def back10(self):
        global frameNum
        frameNum = frameNum - 10
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

        # # Write to transcript .txt file
        # if (str(actions[self.g.get()-1]))=="Idle":
        #     t.write(str(startFrame) + " " + str(frameNum) + " " + str(actions[self.g.get()-1]) + "(" + tool + ")\n")
        # elif self.g.get()==9:
        #     t.write(str(startFrame) + " " + str(frameNum) + " " + str(actions[self.g.get()-1]) + "(L, R, " + str(objects[self.o.get()-1]) + ")\n")
        # elif self.g.get()==10:
        #     t.write(str(startFrame) + " " + str(frameNum) + " " + str(actions[self.g.get()-1]) + "(R, L, " + str(objects[self.o.get()-1]) + ")\n")
        # else:
        #     t.write(str(startFrame) + " " + str(frameNum) + " " + str(actions[self.g.get()-1]) + "(" + tool + ", " + str(objects[self.o.get()-1]) + ")\n")
        t.write(str(startFrame) + " " + str(frameNum) + " " + str(actions[self.g.get()-1]) + "\n")

        # Save image
        # ret, frame = self.vid.get_frame()
        # if ret:
        #     img1 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
        #     imgpil = ImageTk.getimage( img1 )
        #     rgb_im = imgpil.convert('RGB')
        #     rgb_im.save(os.path.join(dir,"Transitions", str(frameNum)+".jpg"), "JPEG" )


        frameNum = frameNum + 1
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
        self.length = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)

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



# Location of dataset folder
dir=os.path.dirname(os.getcwd())

# Get task from command line
try:
    task=sys.argv[1]
    #print(task)
except:
    print("Error: invalid task\nUsage: python gesture_segmentation_labeling.py <task>\nTasks: Suturing, Needle_Passing, Knot_Tying, Pea_on_a_Peg, Post_and_Sleeve, Wire_Chaser_I, Peg_Transfer")
    sys.exit()

# Transcript and video directories
taskDir = os.path.join(dir, "Datasets", "dV", task)
transcriptDir = os.path.join(taskDir,"gestures2")
videoDir = os.path.join(taskDir,"video")

# List of finished transcripts
doneList = [done.split('/')[-1].rsplit(".txt")[0] for done in glob.glob(transcriptDir+'/*.txt')]
#print(doneList)

# Set to run
run = 1

# For each video ending in "_capture1.avi" or ".mp4"
videos = glob.glob(videoDir+"/*.avi") + glob.glob(videoDir+"/*.mp4")
for video in videos:
    #for tool in ["L", "R"]:
        #print(tool)
    trial = video.split("/")[-1]
    # Get name for transcript file name
    trialName = trial.rsplit("_",1)[0]  #+"_"+tool

    if task in ["Pea_on_a_Peg", "Post_and_Sleeve", "Wire_Chaser_I"]:   # single camera videos need different rsplit indexing
        #trialName = trial.rsplit(".")[0]
        trialName = trial.rsplit(".")[0]  #+"_"+tool

    if trialName in doneList:
        continue

    transcriptName = trialName+".txt"
    transcriptPath = os.path.join(transcriptDir, transcriptName)
    print("Working on: "+trialName)

    # Init frameNum counter
    frameNum = 0
    startFrame = 0

    # Create transcript file for this video
    t = open(transcriptPath, 'w')
    #t.write("0 ")

    # Run app
    App(Tk(), trialName, video)

    # Closes all the frames
    t.close()
    cv2.destroyAllWindows()

    # if run == 0:
    #     break
    if run == 0:
        break

        #print(transcriptPath)

# end

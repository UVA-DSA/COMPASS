# Kay Hutchinson 12/8/2021
#
# For labing the videos in the JIGSAWS, DESK, ROSMA, and V-RASTED datasets
# using my new gesture definitions based on context.
#

import os, sys, glob, cv2
from tkinter import *
from tkinter.messagebox import askokcancel, showinfo, WARNING
import PIL
from PIL import Image
from PIL import ImageTk
import pathlib

global frameNum
global run
record = {}

# List of objects and states
objects = ["Nothing", "Ball/Block/Sleeve", "Needle", "Thread", "Fabric/Tissue", "Ring", "Other"]
needleStates = ["Out of", "Touching", "In"]
threadStates = ["Loose", "Taut"]
#cLoopStates = ["Not formed", "Formed"]  # mergeed with knot states 12/8/21
knotStates = ["N/A", "Thread Wrapped", "Loose", "Tight"]
pegStates = ["On", "Off"]
angleStates = ["Open", "Closed"]
peaStates = ["Not held", "In cup", "Stuck together", "Not stuck together", "On peg"]

# Calculate frames per stride using constant
sps = 3  # [strides per second]

# App code
# From https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
class App:
    def __init__(self, window, window_title, video_source=0, labelTarget=None):
        # window is the Tk() object
        global streaming
        streaming = True
        self.window = window
        self.window.title(window_title)
        self.task = window_title[:-8]  # retrieve task name from window_title
        self.video_source = video_source
        self.labelTarget = labelTarget
        print("TASK",self.task)

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        self.frame_stride = int(self.vid.fps * (1.0/sps))

        # Set up layout of widgets
        self.top1 = Frame(window)
        self.top1.pack(side=TOP,expand=True)
        self.top2 = Frame(window)
        self.top2.pack(expand=True)

        self.controls = Frame(window)
        self.controls.pack(expand=True)

        self.progressGroup = Frame(self.controls,pady=10, padx=10)
        self.progressGroup.pack(side=LEFT, expand=True)
        self.grasperGroup = Frame(self.controls)
        self.grasperGroup.pack(side=RIGHT,expand=True)


        self.middle1 = Frame(self.grasperGroup)
        self.middle2 = Frame(self.grasperGroup)
        self.middle3 = Frame(self.grasperGroup)
        self.middle4 = Frame(self.grasperGroup)
        self.middle1.pack(expand=True)
        self.middle2.pack(expand=True)
        self.middle3.pack(expand=True)
        self.middle4.pack(expand=True)


        self.middle5 = Frame(window)
        self.middle5.pack(expand=True)
        self.middle6 = Frame(window)
        self.middle6.pack(expand=True)
        self.middle7 = Frame(window)
        self.middle7.pack(expand=True)
        self.middle8 = Frame(window)
        self.middle8.pack(expand=True)

        self.middleLA = Frame(window)
        self.middleLA.pack(expand=True)
        self.middleRA = Frame(window)
        self.middleRA.pack(expand=True)

        
        self.bottom1 = Frame(window)
        self.bottom1.pack(expand=True)

        self.fowardBackGroup = Frame(window)
        self.fowardBackGroup.pack(expand=True)

        self.forward = Frame(self.fowardBackGroup)
        self.forward.pack(side=RIGHT, expand=True)

        self.backDiv = Frame(self.fowardBackGroup)
        self.backDiv.pack(side=LEFT,expand=True)

        self.bottom2 = Frame(window)
        self.bottom2.pack(expand=True)
        self.bottom3 = Frame(window)
        self.bottom3.pack(side=BOTTOM, expand=True)


        # Note to not use x button to close window
        #self.warning = Text(window, width=37, height=1)
        #self.warning.insert(END,"DO NOT close window, use QUIT button")
        #self.warning.pack(in_=self.top1, side=RIGHT, anchor=E)

        # Create a canvas that can fit the above video source size
        w = int((480.0/self.vid.height)*self.vid.width)
        self.canvas = Canvas(window, width = w, height = 480)  #self.vid.width, height = self.vid.height)
        self.canvas.pack(in_=self.top2, expand=True)


        # Drop down menus for L/R graspers holding/contacting objects
        textBoxWidth = 40
        textBoxWidth2 = 20
        menuBoxWidth = 15
        global frameNum
        self.labelsDone = "190/1900"
        #self.progressLabel = Label(window,width=int(textBoxWidth/2),height=3,textvariable=self.labelsDone, relief=RAISED,pady=10, padx=10, font=10)
        self.progressLabel = Label(
            window,
            wraplength=int(textBoxWidth*2),
            width=int(textBoxWidth/1.5),
            height=3,
            text="Done "+str(int(frameNum/self.frame_stride))+
            " / "+str(int(self.vid.length/self.frame_stride))
            #+" raw:" + str(frameNum)
            ,
            bg='#fff', 
            fg='#006', 
            relief=RAISED,
            pady=10, 
            padx=10, 
            font=20
            )
        #self.progressLabel.config(text="Labeled " + str(int(frameNum/self.frame_stride))+" frames out of "+str(int(self.vid.length/self.frame_stride)))
        
        #text="what's my favorite video?",bg='#fff', fg='#f00',
        #self.progressLabel.set("190/1900")
        self.progressLabel.pack(in_=self.progressGroup)
        self.progressLabel.pack()

        self.LHold = Text(window, width=textBoxWidth, height=1)
        self.LHold.insert(END, "The Left Grasper is holding ")
        self.LHold.pack(in_=self.middle1, side=LEFT)
        self.LH = StringVar(window)
        self.LH.set(objects[0])  # set default
        self.LHoldOpts = OptionMenu(window, self.LH, *objects)
        self.LHoldOpts.config(width=menuBoxWidth)
        self.LHoldOpts.pack(in_=self.middle1, side=RIGHT)

        self.LContact = Text(window, width=textBoxWidth, height=1)
        self.LContact.insert(END, "The Left Grasper is in contact with ")
        self.LContact.pack(in_=self.middle2, side=LEFT)
        self.LC = StringVar(window)
        self.LC.set(objects[0])  # set default
        self.LContactOpts = OptionMenu(window, self.LC, *objects)
        self.LContactOpts.config(width=menuBoxWidth)
        self.LContactOpts.pack(in_=self.middle2, side=RIGHT)

        self.RHold = Text(window, width=textBoxWidth, height=1)
        self.RHold.insert(END, "The Right Grasper is holding ")
        self.RHold.pack(in_=self.middle3, side=LEFT)
        self.RH = StringVar(window)
        self.RH.set(objects[0])  # set default
        self.RHoldOpts = OptionMenu(window, self.RH, *objects)
        self.RHoldOpts.config(width=menuBoxWidth)
        self.RHoldOpts.pack(in_=self.middle3, side=RIGHT)

        self.RContact = Text(window, width=textBoxWidth, height=1)
        self.RContact.insert(END, "The Right Grasper is in contact with ")
        self.RContact.pack(in_=self.middle4, side=LEFT)
        self.RC = StringVar(window)
        self.RC.set(objects[0])  # set default
        self.RContactOpts = OptionMenu(window, self.RC, *objects)
        self.RContactOpts.config(width=menuBoxWidth)
        self.RContactOpts.pack(in_=self.middle4, side=RIGHT)


        # Task-specific context
        if ( "Sutur" in self.task  ) or ( "Needle_P" in self.task ):
            # Needle [not in/in] fabric or ring
            self.Needle = Text(window, width=textBoxWidth2, height=1)
            self.Needle.insert(END, "The needle is ")
            self.Needle.pack(in_=self.middle5, side=LEFT)
            self.N = StringVar(window)
            self.N.set(needleStates[0])
            self.NeedleOpts = OptionMenu(window, self.N, *needleStates)
            self.NeedleOpts.config(width=menuBoxWidth)
            self.NeedleOpts.pack(in_=self.middle5, side=LEFT)
            self.Needle2 = Text(window, width=textBoxWidth2, height=1)
            if ("Sutur" in self.task):
                self.Needle2.insert(END, " the fabric.")
            elif ("Needle_P" in self.task):
                self.Needle2.insert(END, "the ring.")
            self.Needle2.pack(in_=self.middle5, side=RIGHT)

            '''
            # Removed thread states 12/1/21
            # Thread [loose/taut]
            self.Thread = Text(window, width=textBoxWidth2, height=1)
            self.Thread.insert(END, "The thread is ")
            self.Thread.pack(in_=self.middle6, side=LEFT)
            self.T = StringVar(window)
            self.T.set(threadStates[0])
            self.ThreadOpts = OptionMenu(window, self.T, *threadStates)
            self.ThreadOpts.config(width=menuBoxWidth)
            self.ThreadOpts.pack(in_=self.middle6, side=LEFT)
            '''

        elif  ("Knot_Ty" in self.task)  :
            '''
            # C-loop [not formed/formed]
            self.cLoop = Text(window, width=textBoxWidth2, height=1)
            self.cLoop.insert(END, "The c-loop is ")
            self.cLoop.pack(in_=self.middle7, side=LEFT)
            self.C = StringVar(window)
            self.C.set(cLoopStates[0])
            self.cLoopOpts = OptionMenu(window, self.C, *cLoopStates)
            self.cLoopOpts.config(width=menuBoxWidth)
            self.cLoopOpts.pack(in_=self.middle7, side=LEFT)
            '''

            # Knot [loose/tight]
            self.Knot = Text(window, width=2*textBoxWidth2, height=1)
            self.Knot.insert(END, "The status of the knot is ")
            self.Knot.pack(in_=self.middle8, side=LEFT)
            self.K = StringVar(window)
            self.K.set(knotStates[0])
            self.KnotOpts = OptionMenu(window, self.K, *knotStates)
            self.KnotOpts.config(width=menuBoxWidth)
            self.KnotOpts.pack(in_=self.middle8, side=LEFT)


        elif ("Pea_on_" in self.task):
            # Peas stuck together
            self.Peas = Text(window, width=textBoxWidth2, height=1)
            self.Peas.insert(END, "The peas are ")
            self.Peas.pack(in_=self.middle7, side=LEFT)
            self.P = StringVar(window)
            self.P.set(peaStates[0])
            self.PeaOpts = OptionMenu(window, self.P, *peaStates)
            self.PeaOpts.config(width=menuBoxWidth)
            self.PeaOpts.pack(in_=self.middle7, side=LEFT)





        elif ( "Post_an" in self.task):
            # Sleeve on post
            self.Peg = Text(window, width=textBoxWidth2, height=1)
            self.Peg.insert(END, "The sleeve is ")
            self.Peg.pack(in_=self.middle7, side=LEFT)
            self.P = StringVar(window)
            self.P.set(pegStates[0])
            self.PegOpts = OptionMenu(window, self.P, *pegStates)
            self.PegOpts.config(width=menuBoxWidth)
            self.PegOpts.pack(in_=self.middle7, side=LEFT)
            self.Peg2 = Text(window, width=textBoxWidth2, height=1)
            self.Peg2.insert(END, "the post.")
            self.Peg2.pack(in_=self.middle7, side=LEFT)


            
           
        elif ( "Peg_Tr"  in self.task ):
            # peg on pole
            self.Peg = Text(window, width=textBoxWidth2, height=1)
            self.Peg.insert(END, "The peg is ")
            self.Peg.pack(in_=self.middle7, side=LEFT)
            self.P = StringVar(window)
            self.P.set(pegStates[0])
            self.PegOpts = OptionMenu(window, self.P, *pegStates)
            self.PegOpts.config(width=menuBoxWidth)
            self.PegOpts.pack(in_=self.middle7, side=LEFT)
            self.Peg2 = Text(window, width=textBoxWidth2, height=1)
            self.Peg2.insert(END, "the pole.")
            self.Peg2.pack(in_=self.middle7, side=LEFT)



        # Enter button to confirm choice
        self.enter = Button(window,text="Go forward and save label (D)",width=27, font='sans 15',command=self.mark)
        self.enter.pack(in_=self.forward,anchor=CENTER)

        if(not streaming):
            self.done=Button(window,text="Next video", width=15, font='sans 15',command=self.done)
            self.done.pack(in_=self.bottom2, anchor=CENTER)

        # Go Back - Change Choice - Review Labels
        self.back_button = Button(window,text="Go Back and Re-label (A)",width=27 , font='sans 15',command=self.back)
        self.back_button.pack(in_=self.backDiv,anchor=CENTER)
        # watch a couple seconds surrounding the label frame
        self.preview_button = Button(window,text="Preview Clip",width=15 , font='sans 15',command=self.preview)
        self.preview_button.pack(in_=self.bottom2,anchor=CENTER)

        # Done button to finish video
        #if(not streaming):
        

        # Quit button to close properly
        self.quit=Button(window,text="Quit", font='sans 15',command=self.quit)
        self.quit.pack(in_=self.bottom3, anchor=CENTER)

        window.bind("<a>",self.back)
        window.bind("<d>",self.mark)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        # X button to also quit
        window.protocol("WM_DELETE_WINDOW", lambda:quit(window, self.task))
        self.window.mainloop()


    def back(self, etc=False):
        global frameNum
        frameNum = max(frameNum - self.frame_stride,0)
        if( str(frameNum) in record):
            print(record[str(frameNum)])
            self.LHoldOpts.config(bg="#85F280")
            self.LContactOpts.config(bg="#85F280")
            self.RHoldOpts.config(bg="#85F280")
            self.RContactOpts.config(bg="#85F280")
        else:
            self.LHoldOpts.config(bg="#F0F0F0")
            self.LContactOpts.config(bg="#F0F0F0")
            self.RHoldOpts.config(bg="#F0F0F0")
            self.RContactOpts.config(bg="#F0F0F0")
        #self.slider.set(frameNum)
        self.progressLabel.config(
            text="Done " + 
            str(int(frameNum/self.frame_stride))+
            " / "+str(int(self.vid.length/self.frame_stride))
            #+" raw:" + str(int(frameNum))
            )
        self.progressLabel.pack()
        

    def slide(self,f):
        global frameNum
        frameNum = int(f)

    def preview(self):
        #if(hasattr(self, 'preview')):
            # print("already have a preview window up")
            # delete previous
            # start new one at that frame

        prev = Toplevel(self.window)
        self.topPrev = Frame(prev)
        self.topPrev.pack(expand=True)
        prev.title("Context for Frame: " + str(frameNum))

        w = int((480.0/self.vid.height)*self.vid.width)
        self.previewCanvas = Canvas(prev, width = w, height = 480)
        self.previewCanvas.pack(in_=self.topPrev, expand=True)

        self.previewNum = frameNum
        self.previewDistance = -(self.frame_stride*2)

        self.slider = Scale(prev, from_=max(self.previewNum + self.previewDistance,0), length=w, to=(self.previewNum+abs(self.previewDistance)),tickinterval=(self.frame_stride),orient=HORIZONTAL)
        self.slider.set(max(self.previewNum + self.previewDistance,0))
        self.slider.pack(in_=self.topPrev,anchor=CENTER)

        self.prev = prev
        self.updatePreview()

    def updatePreview(self):
        ret, frame = self.vid.get_frame_prev(max(self.previewNum + self.previewDistance,0))
        self.slider.set(max(self.previewNum + self.previewDistance,0))
        nextImg = PIL.Image.fromarray(frame)

        # Scale new image to height of 480 px, keep ratio
        width, height = nextImg.size
        scaling = 480.0/height  # scale to height of 480 px
        newwidth = int(scaling*width)
        nextImg=nextImg.resize((newwidth, 480))

        if ret:
            self.photo_prev = PIL.ImageTk.PhotoImage(image = nextImg)
            self.previewCanvas.create_image(0, 0, image = self.photo_prev, anchor = NW)
        #self.previewDistance = min(self.previewDistance+1,10)
        self.previewStride = self.frame_stride/2
        self.previewDistance +=int(self.frame_stride/4)
        if(self.previewDistance >= self.frame_stride*2):
            self.previewDistance = -(self.frame_stride*2)


        self.prev.after(100, self.updatePreview)

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

    def mark(self, etc=False):
        global frameNum
        if(hasattr(self, "prev")):
            self.prev.destroy()

        frameData = list()
        frameData.append(self.LH.get())
        frameData.append(self.LC.get())
        frameData.append(self.RH.get())
        frameData.append(self.RC.get())

        if ("Sutur" in self.task ) or ( "Needle_" in self.task  ):
            #t.write(str(needleStates.index(self.N.get())) + " ") # needle state
            #t.write(str(threadStates.index(self.T.get())))       # thread state
            frameData.append(self.N.get())
            #frameData.append(self.T.get())   # removed 12/1/21
        elif ( "Knot_Ty" in self.task ):
            #t.write(str(cLoopStates.index(self.C.get())) + " ") # c loop state
            #t.write(str(knotStates.index(self.K.get())))       # knot state
            #frameData.append(self.C.get())
            frameData.append(self.K.get())
        elif ( "Post_a" in self.task) or ( "Peg_Tr"  in self.task ):
        
            frameData.append(self.P.get())
                
        
        elif ("Pea_o" in self.task  ):
            frameData.append(self.P.get())
            

        record[str(frameNum)] = frameData

        '''
        # Old code
        # Write encoded context to transcript .txt file
        t.write(str(frameNum) + " ") # frame number
        t.write(str(objects.index(self.LH.get())) + " " + str(objects.index(self.LC.get())) + " ")  # left grasper
        t.write(str(objects.index(self.RH.get())) + " " + str(objects.index(self.RC.get())) + " ")  # right grasper

        if (self.task == "Suturing") or (self.task == "Needle_Passing"):
            t.write(str(needleStates.index(self.N.get())) + " ") # needle state
            t.write(str(threadStates.index(self.T.get())))       # thread state
        elif self.task == "Knot_Tying":
            t.write(str(cLoopStates.index(self.C.get())) + " ") # c loop state
            t.write(str(knotStates.index(self.K.get())))       # knot state

        t.write("\n")
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

        frameNum = frameNum + self.frame_stride

        if( str(frameNum) in record):
            print(record[str(frameNum)])
            self.LHoldOpts.config(bg="#85F280")
            self.LContactOpts.config(bg="#85F280")
            self.RHoldOpts.config(bg="#85F280")
            self.RContactOpts.config(bg="#85F280")
        else:
            self.LHoldOpts.config(bg="#F0F0F0")
            self.LContactOpts.config(bg="#F0F0F0")
            self.RHoldOpts.config(bg="#F0F0F0")
            self.RContactOpts.config(bg="#F0F0F0")
        # if end of video, close window
        if frameNum > self.vid.length:
            #t.close()
            self.saveToFile()
            self.window.destroy()
        self.progressLabel.config(
            text="Done " + str(int(frameNum/self.frame_stride))+
            " / "+str(int(self.vid.length/self.frame_stride))
            #+" raw:" + str(frameNum)
            )
        self.progressLabel.pack()
        

    def done(self):
        #t.close()
        self.window.destroy()
        self.saveToFile()

    def saveToFile(self):
        self.labelTarget
        print("Saving", self.labelTarget)
        if(os.path.isfile(self.labelTarget)):
            print(self.labelTarget," already present")
            os.remove(self.labelTarget)
        targetBaseName = os.path.dirname(self.labelTarget)
        print("creating folder",targetBaseName )
        if(not os.path.isdir(targetBaseName)):
            path = pathlib.Path(targetBaseName)
            path.mkdir(parents=True, exist_ok=True)

        o = open(self.labelTarget, 'w+')
        row = 0
        #print(record)

        while(str(row) in record):
            data = record[str(row)]
            #print(data)
            o.write(str(row) + " ") # frame number
            o.write(str(objects.index(data[0])) + " " + str(objects.index(data[1])) + " ")  # left grasper
            o.write(str(objects.index(data[2])) + " " + str(objects.index(data[3])) + " ")
            if ( "Suturi" in self.task  ) or ( "Needle_P" in self.task):
                o.write(str(needleStates.index(data[4]))) #+ " ") # needle state
                #o.write(str(threadStates.index(data[5])))       # thread state
            elif ("Knot_Ty" in self.task ):
                #o.write(str(cLoopStates.index(data[4])) + " ")  # c loop state
                o.write(str(knotStates.index(data[4])))         # knot state
            elif ( "Post_an"  in self.task ) or ( "Peg_Tra"  in self.task ):
                o.write(str(pegStates.index(data[4])))
            elif ( "Pea_on_" in self.task  ) :
                o.write(str(peaStates.index(data[4])))

            row+=self.frame_stride
            o.write("\n")
        o.close()
        #self.os.path.join(transcriptDir, transcriptName)
        #record['']
        record.clear()
        #print("Saved")


    def quit(self):
        global run
        # Delete unfinished transcript
        #t.close()
        #os.remove(t.name)
        self.window.destroy()
        run = 0

def quit(window, task):
        answer = askokcancel(
            title='Exiting labeling attempt for ' + task,
            message='Labels for this video will be discarded.',
            icon=WARNING)
        if answer: 
            global run
            # Delete unfinished transcript
            #t.close()
            #os.remove(t.name)
            window.destroy()
            run = 0
        '''
        close = askokcancel("Close", "Would you like to close the program?")
        if close:
            
            global run
            # Delete unfinished transcript
            #t.close()
            #os.remove(t.name)
            window.destroy()
            run = 0
        '''

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
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))

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
            print("get_frame(self) undefined variable \"ret\"")
            return (ret, None)

    def get_frame_prev(self, previewFrame):
        if self.vid.isOpened():
            self.vid.set(1, previewFrame)
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
            print("get_frame(self) undefined variable \"ret\"")
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# MAIN -------------------------------------------------------------------------
# Location of dataset folder
dir=os.path.dirname(os.getcwd())
all_tasks = ["Suturing", "Needle_Passing", "Knot_Tying", "Pea_on_a_Peg", "Post_and_Sleeve", "Wire_Chaser_I", "Peg_Transfer", "Multiple"]
#top_dir = os.path.join(dir, "Datasets")
task = "Peg_Transfer" # default
# Get task from command line
try:
    task=sys.argv[1]
    #print(task)
except:
    print("Error: invalid task\nUsage: python gesture_segmentation_labeling.py <task>\nTasks: Suturing, Needle_Passing, Knot_Tying, Pea_on_a_Peg, Post_and_Sleeve, Wire_Chaser_I, Peg_Transfer")
    #sys.exit()

'''
for root, dirs, files in os.walk(top_dir):
    for d in dirs:
        try:
            b=all_tasks.index(d)
            print(d)
            task = d
        except ValueError:
            "do nothing"
'''


#task="Pea_on_a_Peg"
# or Suturing

# done Peg Transfer
# 

# Transcript and video directories
taskDir = os.path.join(dir, "Datasets", "dV", task)
transcriptDir = os.path.join(taskDir,"transcriptions_context")
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
videos = glob.glob(videoDir+"\\*.avi") + glob.glob(videoDir+"\\*.mp4")
non_empty = False
for video in videos:
    
    trial = video.split("\\")[-1]
    # Get name for transcript file name
    trialName = trial.rsplit("_",1)[0]

    if task in ["Pea_on_a_Peg", "Post_and_Sleeve", "Wire_Chaser_I"]:   # single camera videos need different rsplit indexing
        #trialName = trial.rsplit(".")[0]
        trialName = trial.rsplit(".")[0]

    if trialName in doneList:
        continue
    
    transcriptName = trialName+".txt"
    transcriptPath = os.path.join(transcriptDir, transcriptName)

    # Init frameNum counter
    frameNum = 0
    startFrame = 0

    # Create transcript file for this video
    #t = open(transcriptPath, 'w')
    #t.write("0 ")
    
    # Run app
    non_empty = True
    App(Tk(), trialName, video, transcriptPath)

    # Closes all the frames
    #t.close()
    cv2.destroyAllWindows()

    if run == 0:
        break


    #print(transcriptPath)
if(not non_empty):
    root = Tk()
    answer = askokcancel(
                title='No unlabeled videos found.',
                message='Check that the transcriptions folder contains a file for every video. If so, you\'re finished and can send the trasncriptions back.',
                icon=WARNING)
    if answer: 
            root.destroy()
            run = 0

# end
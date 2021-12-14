import array
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as signal
from scipy.fftpack import fft as FFT


class UAEdata(object):
   def __init__(my,sequence="H",speed=7,yaw=10,UAEpath=None,rep=None):
      if UAEpath is None:
         UAEpath = "/Users/tnakamoto/Desktop/atlantis/datasets/nrel-data/"
      
      if speed >=10:
         speedstring  = str(speed)
      else:
         speedstring = "0" + str(speed)
      if yaw >= 10:
         yawstring = str(yaw)
      else:
         yawstring = "0" + str(yaw)
      if rep is None: #Repetition number (usually 0, some data entries have multiple collections)
         rep = "0"
      else:
          rep=str(rep)
      if sequence=="H":
         my.dataFile =UAEpath+"uae6."+"z07.00.h"+speedstring+"00"+yawstring+rep+".eng"
         my.headerFile= UAEpath+"uae6."+"z07.00.h"+speedstring+"00"+yawstring+rep+".hd1"
      elif sequence=="I":
         my.dataFile =UAEpath+"uae6."+"z08.00.i"+speedstring+"00"+yawstring+rep+".eng"
         my.headerFile= UAEpath+"uae6."+"z08.00.i"+speedstring+"00"+yawstring+rep+".hd1"

      else:
         print("Haven't implemented for other sequences yet")
         quit()
   
      print("Identified", my.headerFile)
      print("Identified", my.dataFile)


      my.sequence=sequence # UAE data sequence
      my.UAEpath=UAEpath         # Location of UAE data directory
      my.speed=speed       # Inlet speed specification
      my.yaw=yaw           # Yaw angle specification
      my.rep=rep           # Repetition integer
      my.data, my.channels, my.codes, my.names, my.units= my.readHeader(my.UAEpath) #Channel headerFile information
      my.v = {}            #Dictionary of data variables of interest
      my.extractdata()     #Gets data  and puts them in v{}
      my.bladelength=  5.029      #TODO: This is the current StarCCM value, which is too long. Gotta fix that.
      xuae = np.array([.3,.47,.63,.80,.95])*my.bladelength
      my.xuae=xuae   #Blade spanwise positions for which load data have been collected
      
   def bychannel(my,channel):
      # This function gets the index from a channel number designation
      for  n,val  in enumerate(my.channels): 
         print(n,val)
         if channel == val:   
            return n
         
      
   def readHeader(my, displaytext=False):
      headerFile=my.headerFile
      dataFile=my.dataFile

      print("Reading HF %s"%dataFile )
      #Declare the main variables
      channels=[];codes = [];names =[];units =[]
      with open(headerFile) as FID:
         lines = FID.readlines()
         line = lines[1]
         commalocs = [c for c in range(len(line))  if line[c] ==","]
         nchan = int(line[0:commalocs[0]])
         nsamp  = int(line[commalocs[0]+1:commalocs[1]])
         for i in range(2,len(lines)):
            line = lines[i]
            commalocs = [c for c in range(len(line)) if line[c]==","]
            channels.append(line[0:commalocs[0]])
            codes.append(line[commalocs[0]+1:commalocs[1]])
            names.append(line[commalocs[1]+1:commalocs[2]])
            units.append(line[commalocs[4]+1:commalocs[5]])
         
         
         data =  my.loadDataFile(dataFile ,nchan,nsamp)
      return data, channels, codes, names, units
      
   
   def loadDataFile(my,dataFile ,nchan,nsamp):
      with open(dataFile ,'rb') as fid: # little endian ieee-le
         data = np.zeros([nsamp,nchan])   
         errbit = np.zeros(nsamp)
         for i in range(nsamp):
            temp = np.fromfile(fid,dtype='<f',count=nchan)
            data[i,:]=temp
            temp = np.fromfile(fid,dtype=np.uint8,count=1)
            errbit[i] = temp
         return data
   
   @staticmethod
   def elapsedtime(hour,minute,second,millisec):
      hour = hour *3600.0
      mins = minute*60.0
      sec = second
      milli = millisec/1000.0
      time = hour + mins+sec+milli
      time=time-time[0]
      return time

   def extractdata(my,chordfit=None,twistfit=None):
      d = my.data
      # Looking at a NREL data file, subtract three from the line number
      # to get the index. Line 206 is yaw angle, corresponds to index 203
      bend_flat_root_aero = d[:,220]
      bend_flat_root_teet = d[:,221]
      time= np.array(my.elapsedtime(d[:,207],d[:,209],d[:,211],d[:,212])) #Converts to seconds
      power = np.array(d[:,225])
      turbyaw = np.array(d[:,203])

      #Normal force and tangential force from coefficients
      cn    = np.array([d[:,227],d[:,233],d[:,239],d[:,245],d[:,251]]) # Normal force coefficient
      ct    = np.array([d[:,228],d[:,234],d[:,240],d[:,246],d[:,252]]) # Tangential force coefficient
      cth   = np.array([d[:,229],d[:,235],d[:,241],d[:,247],d[:,253]]) # Local thrust coefficient
      ctq   = np.array([d[:,230],d[:,236],d[:,242],d[:,248],d[:,254]]) # Local torque coefficient
      qnorm = np.array([d[:,232],d[:,238],d[:,244],d[:,250],d[:,256]]) # Local Dynamic Pressure
      wtatemp = np.array(d[:,186])
      wtapress= np.array(d[:,184])
      azim = np.array(d[:,201])
      etorque = np.array(d[:,219]) # Estimated Torque
      ethrust = np.array(d[:,218]) # Estimated Thrust
      rpm = np.array(d[:,216])
  
      look = turbyaw
      c = 0 #273.15 if K
      #print(np.mean(look)+c)
      #print(np.min(look)+c)
      #print(np.max(look)+c)
      
      # Force = Coeff * DynPressure * Chord
      chords = np.array([.711,.627,.542,.457,.381])
      ftan  = (ctq.T * qnorm.T * chords).T
      fnorm = (cth.T * qnorm.T * chords).T
      thrust=ethrust
      torque=etorque
      
      my.v["WTtemperature"] = np.array(d[:,186])
      my.v["WTpressure"] = np.array(d[:,182])
      my.v["WTrho"] = np.array(d[:,276])
      my.v["WTV"] = np.array(d[:,274])
      my.v["cone"] = np.array(d[:,215])
      my.v["rpm"] = np.array(d[:,216])

      my.v["azAngle"] = azim
      my.v["cn"] = cn
      my.v["ct"] = ct
      my.v["time"] = time
      my.v["fnorm"] = fnorm
      my.v["ftan"]  = ftan
      my.v["thrust"] = thrust
      my.v["torque"] = torque
      my.v["lssTorque"]=np.array(d[:,158])
      return cn, ct, time, fnorm, ftan, thrust, torque
   
   def plotAzAngleVsLoads(my,N=0):
      az = my.v["azAngle"]
      fx = my.v["fnorm"][N]
      fy = my.v["ftan"][N]
      # print(my.v["ftan"].shape)
      plt.plot(az,fx,marker="o",linestyle="None",label="normal")
      plt.plot(az,fy,marker="o",linestyle="None",label="tangential")
      plt.hlines([np.mean(fx),np.mean(fy)],0,360,color="blue",label="Exp Mean",zorder=100)
      plt.legend()
   
   def plotUnsteadyLoading(my):
      az = my.v["azAngle"]
      thrust=my.v["thrust"]
      torque=my.v["torque"]
      plt.plot(az,thrust,marker="o",linestyle="None",label="thrust")
      plt.plot(az,torque,marker="o",linestyle="None",label="torque")
      plt.legend()
   
   def signalProc(my,PLOT=False):
      # This could be rewritten so that it can be used to analyze any of the variables
      # instead of strictly the thrust and torque (not both at the same time)
      t=my.v["time"]

      thrust=my.v["thrust"]
      torque=my.v["torque"]
      
      # FFT of the varibles of interest
      ffta = FFT(thrust-np.mean(thrust))
      fftb = FFT(torque-np.mean(torque))
      
      # Amplitude here is taken as the difference between highest and lowest points
      thrustAmp=(np.max(thrust)-np.min(thrust))/2
      torqueAmp=(np.max(torque)-np.min(torque))/2

      

      # print(np.mean(thrust))
      # print(np.min(thrust))
      # print(np.max(thrust))

      if PLOT:
         # This produces quite a bit of plot. And I'm not sure I really want all of that.
         fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
         n=np.size(t)
         Fs=np.size(t)/np.max(t)
         print(Fs)
         print(np.size(t))
         print(np.max(t))
         fr=Fs/2*np.linspace(0,1,int(n/2))

         y_m=2/n*abs(ffta[0:np.size(fr)])
         y_m2=2/n*abs(fftb[0:np.size(fr)])
         ax[0].plot(t,thrust,label="thrust")
         ax[0].hlines([np.min(thrust),np.max(thrust)],np.min(t),np.max(t),color='k')
         ax[0].hlines([np.min(torque),np.max(torque)],np.min(t),np.max(t),color='k')
         ax[0].plot(t,torque,label="torque")
         ax[0].legend()
         ax[0].set_xlabel("Time (s)")
         ax[0].set_ylabel("Load magnitude")
         ax[1].stem(fr*60,y_m,label="thrust")
         ax[1].stem(fr*60,y_m2,label="torque",markerfmt="C1o",linefmt="C1-")
         plt.xlim([0,1000])
         ax[1].legend()

         ax[1].set_xlabel("Frequency (Cycles per Minute)")
         ax[1].set_ylabel("Amplitude")
      



if __name__ == "__main__":
   case = 1

   if case == 1:
      speed=7; yaw=10
      a=UAEdata(speed=speed,yaw=yaw)
      a.signalProc(PLOT=True)
      plt.show()
      quit()

   elif case == 2:
      # Here we save the unsteady loading from a bunch of cases
      speeds = [7,10,15]
      yaws = [0,10,30,60]
      for speed in speeds:
         for yaw in yaws:
            a=UAEdata(speed=speed,yaw=yaw)
            a.plotUnsteadyLoading()
            plt.savefig("figs/"+str(speed)+"."+str(yaw)+".png")
            plt.clf()
      quit()
   
   elif case == 3:
      # Here we use plotAzAngleVsLoads
      a=UAEdata(speed=7,yaw=60)
      for i in range(5):
         a.plotAzAngleVsLoads(N=i)
         plt.show()
      a.plotUnsteadyLoading()
      plt.show()
      quit()

   elif case == 4:
      # Here we plot loads at particular spanwise locations by grabbing output data 
      # that we've pulled into the main scope
      print("TESTING")
      test=UAEdata(speed=7,yaw=10)
      cn, ct, time, fnorm, ftan, thrust, torque = test.extractdata()
      
      for i in range(len(cn)):
         plt.plot(time,ct[i],label="ct")
         plt.plot(time,cn[i],label="cn")
         plt.legend()
         plt.show()
      quit()




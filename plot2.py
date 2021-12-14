import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# from postproccer import UAEsim,Graph,UAEdata
from UAEdata import UAEdata
class Simulation(object):
   def __init__(my, path=None):
      if path is None:
         print("Specify path"); quit()
      else:
         my.path=path
      my.importData()

   def specifyVariable(my,name):
      if name == "time":
         return 'Physical Time: Physical Time (s)'
      elif name == "thrustForce":
         return 'Thrust 1 Monitor: Thrust (N)'
         #return 'Total Blade Thrust (Force) Monitor: Force (N)'	
      elif name == "thrustThrust":
         return 'Thrust 1 Monitor: Thrust (N)'
         #return'Total Blade Thrust (Thrust) Monitor: Thrust (N)'	
      elif name == "torque":
         return 'Moment 1 Monitor: Moment (N-m)'
         #return 'Total Blade Torque (Moment) Monitor: Moment (N-m)'
      elif name == "tandi":
         return 'Tangential_%i Monitor: Force (N)'
      elif name == "normdi":
         return 'Normal_%i Monitor: Force (N)'
      else:
         print("Need to look at specifyVariable call")
         quit()
   
   def importData(my):
      df = pd.read_csv(my.path,sep=',')
      print()
      print(df.shape)
      print(len(df.columns))
      print(len(df))

      time='Physical Time: Physical Time (s)'
      thrustForce='Thrust 1 Monitor: Thrust (N)'
      #thrustForce='Total Blade Thrust (Force) Monitor: Force (N)'	
      #thrustThrust='Total Blade Thrust (Thrust) Monitor: Thrust (N)'	
      thrustThrust='Thrust 1 Monitor: Thrust (N)'
      # torque='Total Blade Torque (Moment) Monitor: Moment (N-m)'
      torque='Moment 1 Monitor: Moment (N-m)'
      tandi= 'Tangential_%i Monitor: Force (N)'
      normdi='Normal_%i Monitor: Force (N)'


      Tthrust = df[thrustThrust] #Total thrust as a function of time (thrust report)
      Fthrust = df[thrustForce]  #Total thrust as a funciton of time (force report)
      timevec = df[time]         #Vector of real time steps
      my.data=df
      my.timevec=timevec

   def plotTimeSeries(my,varname,**kwargs):
      time=my.data[my.specifyVariable("time")]
      var = my.data[my.specifyVariable(varname)]
      plt.plot(time,var,**kwargs)
      return(time,var)
      #plt.xlabel("Time (s)")



class ForceDistribution():
   def __init__(my,filname=None,tag=None,firstlast=True):
      time='Physical Time: Physical Time (s)'
      df = pd. read_csv(filname,sep=',')
      nt = len(df[time])
      # if firstlast is True:
      #    FD = np.zeros((46+1,nt))
      #    for i in range(1,46+1):
      #       FD[i,:]=df[tag%(i)].values/0.1 
      #    print(FD)
         
      FD = np.zeros((46+1,nt))
      for i in range(1,46):
         FD[i-1,:]=df[tag%(i+1)].values/0.1 
      FD[46,:]=df[tag%(1)].values/0.1
      # FD = np.zeros((46+1,nt))
      # for i in range(1,46+1):
      #    FD[i,:]=df[tag%(i)].values/0.1 
        

      my.FD=FD

def plotUAE(tan=False,norm=False,V=7.0,yaw=0.0,sequence="H"):
   blade=[5.029]
   temp = UAEdata(speed=int(V),yaw=int(yaw),sequence=sequence)  #TODO: Fix Yaw, path
   cn,ct,time,fx,fy,thrust,torque = temp.extractdata()
   xuae = np.array([.3,.47,.63,.80,.95])*max(blade)

   labeltext = "UAE Data Range (mean, min, max)"

   if tan is True:
      # Mean
      yuae2= [np.mean(f) for f in fy]
      plt.plot(xuae,yuae2,marker="o",linewidth=0.0,markersize=6,color='k',label=labeltext)
      # Max
      yuae2= [np.max(f) for f in fy] 
      plt.plot(xuae,yuae2,marker="o",linewidth=0.0,markersize=6,color='k')
      # Min
      yuae2= [np.min(f) for f in fy]
      plt.plot(xuae,yuae2,marker="o",linewidth=0.0,markersize=6,color='k')

   if tan is True and norm is True:
      labeltext=None

   if norm is True:
      # Mean
      yuae = [np.mean(f) for f in fx]
      plt.plot(xuae,yuae,marker="o",linewidth=0.0,markersize=6,label=labeltext,color='k')
      # Max
      yuae = [np.max(f) for f in fx]
      plt.plot(xuae,yuae,marker="o",linewidth=0.0,markersize=6,color='k')
      # Min
      yuae = [np.min(f) for f in fx]
      plt.plot(xuae,yuae,marker="o",linewidth=0.0,markersize=6,color='k')

if __name__ == "__main__":
   files=[
   'Coarse_Copy_Data.csv',       #0 
   'Coarse_Data.csv',            #1
   'Data_20.csv',                #2
   'Data_30.csv',
   'Data_40Its.csv',
   'Data_2blade_40.csv',         #5
   'Data_2blade_30.csv',
   'Data_2blade_20.csv',
   'Data_Copy_CT.csv',           #8
   'Data_1blade_20_5400.csv',
   'Data_2blade_LE.csv',
   'Data_2blade_LE2.csv',
   'Data_1blade_30_5400.csv',    #12
   'Data_2blade_LE2_30.csv',     #13   
   'Data_2blade_LE2_HT.csv',
   'Data_H_C_10deg.csv',
   'Data_H_C_1degCONT.csv',
   'Data_H_C_1degCONT2.csv',
   'Data_H_C_05degCONT.csv',     #18
   'Data_H_05deg1Rot.csv',       #19
   'Data_H_10y_1deg_1rot.csv',
   'Data_H_45y_1deg1rot.csv'    #21

   ]

   


   A=Simulation(files[0])
   A.plotTimeSeries("thrustThrust",label="Coarse mesh",linestyle="--")

   B=Simulation(files[9])
   B.plotTimeSeries("thrustThrust",label="thrust report 1 blade fine time")
   B=Simulation(files[10])
   B.plotTimeSeries("thrustThrust",label="thrust report 2 blade LE refine")
   B=Simulation(files[13])
   B.plotTimeSeries("thrustThrust",label="thrust report 2 blade LE refine")
   B=Simulation(files[14])
   B.plotTimeSeries("thrustThrust",label="thrust report 2 blade LE refine 1/2 time")
   # B.plotTimeSeries("thrustThrust",label="normal force report (unyawed)")
   B=Simulation(files[7])
   B.plotTimeSeries("thrustForce",label="normal thrust report fine 2blade mesh")
   
   
   B=Simulation(files[7])
   B.plotTimeSeries("torque",label="torque report fine 2blade mesh")

   B=Simulation(files[9])
   B.plotTimeSeries("torque",label="torque report 1blade fine time")
   B=Simulation(files[10])
   B.plotTimeSeries("torque",label="torque report 2blade LE refined")
   B=Simulation(files[13])
   B.plotTimeSeries("torque",label="torque report 2blade LE refined 30")
   B=Simulation(files[14])
   B.plotTimeSeries("torque",label="torque report 2blade LE refined 1/2")

   C= UAEdata(speed=7,yaw=0)
   plt.plot(C.v["time"],C.v["thrust"]/2,label="UAE Unyawed thrust")
   plt.plot(C.v["time"],C.v["torque"]/2,label="UAE Unyawed torque")

   #plt.plot(timevec,Tthrust,label="thrust report")
   #plt.plot(timevec,Fthrust,label="force report")
   plt.xlabel("Time")
   plt.ylabel("Total normal force")
   plt.legend()
   plt.show()
   # plt.quit()

   A=Simulation(files[0])
   time,torqA= A.plotTimeSeries("torque",label="normal thrust report Improved")
   # A.plotTimeSeries("thrustThrust",label="normal force report (10deg)")

   B=Simulation(files[1])
   _,torqB=B.plotTimeSeries("torque",label="normal thrust report Poor Transition")
   # B.plotTimeSeries("thrustThrust",label="normal force report (unyawed)")
   plt.close()
   plt.plot(time,torqA*-2,label="torque with improved transition")
   plt.plot(time,torqB*-2,label="torque with poor transition")
   C= UAEdata(speed=7,yaw=0)
   plt.plot(C.v["time"],C.v["torque"]/2,label="UAE Unyawed")
   plt.plot(C.v["time"],C.v["lssTorque"]/2,label="UAE LSS Torque")
   #plt.plot(timevec,Tthrust,label="thrust report")
   #plt.plot(timevec,Fthrust,label="force report")
   plt.xlabel("Time")
   plt.ylabel("Total torque")
   plt.legend()
   plt.show()
   # quit()

   midpoints = [0.508 + 0.05 + 0.1 * n for n in range(47) ]
   print(len(midpoints))

   time_examine=134


   tandi= 'Tangential_%i Monitor: Force (N)'
   normdi='Normal_%i Monitor: Force (N)'

   # Tan_naive = ForceDistribution(filname=files[1],tag=tandi)
   # plt.plot(midpoints,Tan_naive.FD[:,time_examine])

   # Tan_prism = ForceDistribution(filname=files[3],tag=tandi)
   # plt.plot(midpoints,Tan_prism.FD[:,time_examine])

   # Norm_naive = ForceDistribution(filname=files[1],tag=normdi)
   # plt.plot(midpoints,Norm_naive.FD[:,time_examine])

   # Norm_prism = ForceDistribution(filname=files[3],tag=normdi)
   # plt.plot(midpoints,Norm_prism.FD[:,time_examine])

   

   # Norm_prism = ForceDistribution(filname=files[1],tag=normdi)
   # plt.plot(midpoints,Norm_prism.FD[:,time_examine],label="Poor")

   """ Norm_prism = ForceDistribution(filname=files[2],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,time_examine],label="Fine 1blade",linestyle=":")

   Norm_prism = ForceDistribution(filname=files[3],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,time_examine],label="Mid 1blade",linestyle=":")
   Norm_prism = ForceDistribution(filname=files[0],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,time_examine],label="Coarse 1blade",linestyle=":")

   # Norm_prism = ForceDistribution(filname=files[4],tag=normdi)
   # plt.plot(midpoints,Norm_prism.FD[:,time_examine],label="40 m 40 Its")

   
   Norm_prism = ForceDistribution(filname=files[7],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,time_examine],label="Fine 2blade",linestyle="--")
   Norm_prism = ForceDistribution(filname=files[6],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,time_examine],label="Mid 2blade",linestyle="--")
   Norm_prism = ForceDistribution(filname=files[5],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,time_examine],label="Coarse 2blade",linestyle="--")

   Norm_prism = ForceDistribution(filname=files[12],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,int(time_examine*2)],label="Mid 1blade 1/2 time",linewidth=4)
   # Norm_prism = ForceDistribution(filname=files[9],tag=normdi)
   # plt.plot(midpoints,Norm_prism.FD[:,int(time_examine*2)],label="Coarse 1/2 time")
   

   Norm_prism = ForceDistribution(filname=files[10],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,int(time_examine)],label="Coarse 2blade LE",linestyle="-")
   Norm_prism = ForceDistribution(filname=files[11],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,int(time_examine)],label="Coarse 2blade LE2",linestyle="-",linewidth=2)
   Norm_prism = ForceDistribution(filname=files[13],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,int(time_examine)],label="Mid 2blade LE2",linestyle="-",linewidth=2)
   Norm_prism = ForceDistribution(filname=files[14],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,int(329)],label="Coarse 2blade  LE 2 1/2 time",linewidth=3)
   
   
   # Norm_prism = ForceDistribution(filname=files[5],tag=normdi)
   # plt.plot(midpoints,Norm_prism.FD[:,time_examine],label="wall model")
   # Norm_prism = ForceDistribution(filname=files[7],tag=normdi)
   # plt.plot(midpoints,Norm_prism.FD[:,time_examine],label="Large Rotate")

   # Norm_prism = ForceDistribution(filname=files[8],tag=normdi)
   # plt.plot(midpoints,Norm_prism.FD[:,time_examine],label="Large Rotate 10deg yaw")

   plt.legend()
   plt.ylim((0,300))
   # plotUAE(norm=True)
   plotUAE(norm=True, sequence="I")
   plt.show()

   # Coarse Mid Fine corresponds to 40, 30, 20 base size meshes
   Tan_prism = ForceDistribution(filname=files[1],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,time_examine],label="Coarse 1blade")


   # Tan_prism = ForceDistribution(filname=files[0],tag=tandi)
   # plt.plot(midpoints,Tan_prism.FD[:,time_examine]*midpoints,label="Poor")
   Tan_prism = ForceDistribution(filname=files[2],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,time_examine],label="Fine 1blade ")
   Tan_prism = ForceDistribution(filname=files[3],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,time_examine],label="Mid 1blade")
   # Tan_prism = ForceDistribution(filname=files[4],tag=tandi)
   # plt.plot(midpoints,Tan_prism.FD[:,time_examine]*midpoints,label="40 m 40 its")
   Tan_prism = ForceDistribution(filname=files[5],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,time_examine],label="Coarse 2blade",linestyle="--")
   Tan_prism = ForceDistribution(filname=files[6],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,time_examine],label="Mid 2blade",linestyle="--")
   Tan_prism = ForceDistribution(filname=files[7],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,time_examine],label="Fine 2blade",linestyle="--")
   # Tan_prism = ForceDistribution(filname=files[8],tag=tandi)
   # plt.plot(midpoints,Tan_prism.FD[:,time_examine]*midpoints,label="40 1blade CorrThermo")
   # Tan_prism = ForceDistribution(filname=files[9],tag=tandi)
   # plt.plot(midpoints,Tan_prism.FD[:,int(time_examine*2)]*midpoints,label="Coarse 1/2 Time")
   Tan_prism = ForceDistribution(filname=files[12],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,int(time_examine*2)],label="Mid 1/2 Time")
   Tan_prism = ForceDistribution(filname=files[10],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,int(time_examine)],label="Coarse 2blade LE",linestyle=":")
   Tan_prism = ForceDistribution(filname=files[11],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,int(time_examine)],label="Coarse 2blade LE2",linestyle=":")
  
   Tan_prism = ForceDistribution(filname=files[13],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,int(time_examine)],label="Mid 2blade LE2",linestyle=":")
   
   Tan_prism = ForceDistribution(filname=files[14],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,int(329)],label="Mid 2blade LE2 1/2 Time",linewidth=3)

   

   plt.ylim((0,120))
   plt.legend()
   plotUAE(tan=True,sequence="I")
   plt.show()



   
   Norm_prism = ForceDistribution(filname=files[15],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,int(160)],label="H Coarse 10deg Thrust (5rots)",linewidth=3)
   
   Tan_prism = ForceDistribution(filname=files[15],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,int(160)],label="H Coarse 10deg Torque (5rots)",linewidth=3,linestyle="--")

   Norm_prism = ForceDistribution(filname=files[16],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,int(539)],label="H Coarse 1deg Thrust (1rot)",linewidth=3)
   
   Norm_prism = ForceDistribution(filname=files[17],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,int(899)],label="H Coarse 1deg Thrust (2rot)",linewidth=3)

   Norm_prism = ForceDistribution(filname=files[18],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,int(899+360)],label="H Coarse 0.5deg Thrust (1/2rot)",linewidth=3)

   Norm_prism = ForceDistribution(filname=files[19],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,int(899+360+360)],label="H Coarse 0.5deg Thrust (1rot)",linewidth=3)

   Tan_prism = ForceDistribution(filname=files[16],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,int(539)],label="H Coarse 1deg Torque (1rot)",linewidth=3,linestyle="--")
   
   Tan_prism = ForceDistribution(filname=files[17],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,int(899)],label="H Coarse 1deg Torque (2rot)",linewidth=3,linestyle="--")

   Tan_prism = ForceDistribution(filname=files[18],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,int(899+360)],label="H Coarse 0.5deg Torque (1/2rot)",linewidth=3,linestyle="--")

   Tan_prism = ForceDistribution(filname=files[19],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,int(899+360+360)],label="H Coarse 0.5deg Torque (1rot)",linewidth=3,linestyle="--")




   plt.legend()
   plt.ylim((0,300))
   plotUAE(tan=True,sequence="H")
   plotUAE(norm=True, sequence="H")
   plt.show()


   B=Simulation(files[19])
   B.plotTimeSeries("torque",label="torque report 2blade LE refined 1/2")
   B.plotTimeSeries("thrustThrust",label="torque report 2blade LE refined 1/2")

   

   C= UAEdata(speed=7,yaw=0)
   plt.plot(C.v["time"],C.v["thrust"],label="UAE Unyawed thrust")
   plt.plot(C.v["time"],C.v["torque"],label="UAE Unyawed torque")
   plt.show()


  

   Norm_prism = ForceDistribution(filname=files[20],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,int(160)],label="H Coarse 10yaw Thrust  10deg(5rots)",linewidth=3)

   Norm_prism = ForceDistribution(filname=files[20],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,int(160+360)],label="H Coarse 10yaw Thrust 1deg(1rots)",linewidth=3)
   
   Tan_prism = ForceDistribution(filname=files[20],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,int(160)],label="H Coarse 10yaw Torque 10deg(5rots)",linewidth=3,linestyle="--")

   Tan_prism = ForceDistribution(filname=files[20],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,int(160+360)],label="H Coarse 10yaw Torque 1deg(1rots)",linewidth=3,linestyle="--")


   C=UAEdata(speed=7,yaw=10)
   plotUAE(tan=True,sequence="H",yaw=10)
   plotUAE(norm=True, sequence="H",yaw=10)
   plt.legend()
   plt.ylim((0,300))
   plt.show()
   """


   Norm_prism = ForceDistribution(filname=files[21],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,int(170)],label="H Coarse 45yaw Thrust  10deg(5rots)",linewidth=3)

   Norm_prism = ForceDistribution(filname=files[21],tag=normdi)
   plt.plot(midpoints,Norm_prism.FD[:,int(170+360)],label="H Coarse 45yaw Thrust 100deg(1rots, oops)",linewidth=3)
   
   Tan_prism = ForceDistribution(filname=files[21],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,int(170)],label="H Coarse 45yaw Torque 10deg(5rots)",linewidth=3,linestyle="--")

   Tan_prism = ForceDistribution(filname=files[21],tag=tandi)
   plt.plot(midpoints,Tan_prism.FD[:,int(170+360)],label="H Coarse 45yaw Torque 100deg(1rots, oops)",linewidth=3,linestyle="--")


   C=UAEdata(speed=7,yaw=45)
   plotUAE(tan=True,sequence="H",yaw=45)
   plotUAE(norm=True, sequence="H",yaw=45)
   plt.legend()
   plt.ylim((0,300))
   plt.show()






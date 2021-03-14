import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import statistics
import seaborn as sb



df = pd.read_csv("time5.csv")
df["Num_Iterations"]=df["Iterations"]*df["Ghostcells"]
df1=df.query('N == 10000 and Iterations == 100 and Output == False and Ghostcells==10')



df6=df.query('Num_Iterations==1000 and N == 10000 and Output == False and Processes==32')
df5=df.query('Num_Iterations==1000 and N == 10000 and Output == False and Processes==36')
df3=df.query('Num_Iterations==1000 and N == 10000 and Output == False and Processes==37')
plt.figure(1)

x=df6["Ghostcells"]
y=df6["Overall_Time"]
new_x, new_y = zip(*sorted(zip(x, y)))

plt.plot(new_x,new_y,color="green")
plt.plot(new_x,new_y,"o",label="1D Implementierung (32 Prozesse)",color="green")

plt.suptitle("Gesamtlaufzeit in Abbhängigkeit von der Geisterzellenbreite und Implementierungsart", fontsize=10)
plt.title("N: 10000, α=0.000001, Iterations=1000 ", fontsize=8)

x=df5["Ghostcells"]
y=df5["Overall_Time"]
new_x, new_y = zip(*sorted(zip(x, y)))

plt.plot(new_x,new_y,color="darkblue")
plt.plot(new_x,new_y,"o",color="darkblue",label="2D Implementierung (36 Prozesse)")
plt.xlabel('Geisterzellenbreite')
plt.ylabel('Overall Time')

x=df3["Ghostcells"]
y=df3["Overall_Time"]
new_x, new_y = zip(*sorted(zip(x, y)))

plt.plot(new_x,new_y,color="lightblue")
plt.plot(new_x,new_y,"o",color="lightblue",label="1D Implementierung (37 Prozesse)")
plt.legend()

plt.savefig("Ghostcells_10000.png")


df5=df.query('Num_Iterations==1000 and N == 1000 and Output == False and Processes==36')




#figure2

plt.figure(2)
plt.suptitle("Gesamtlaufzeit in Abbhängigkeit von der Geisterzellen", fontsize=10)
plt.title("N: 10000, α=0.000001, Iterations=100, Prozesse=36 ", fontsize=8)

#ax=sb.lmplot(x="Ghostcells", y="Overall_Time", data=df5, ci=None, lowess=True)
x=df5["Ghostcells"]
y=df5["Overall_Time"]
new_x, new_y = zip(*sorted(zip(x, y)))

plt.plot(new_x,new_y,color="green")
plt.plot(new_x,new_y,"o",label="2D Implementierung (36 Prozesse)",color="green")

plt.savefig("ghostcells_1000_1D.png")


df5=df.query('Num_Iterations==1000 and N == 1000 and Output == False and Processes==32')
df6=df.query('Num_Iterations==1000 and N == 1000 and Output == False and Processes==37')

plt.suptitle("Gesamtlaufzeit in Abbhängigkeit von der Geisterzellen und Implementierungsart", fontsize=10)
plt.title("N: 1000, α=0.000001, Iterations=1000", fontsize=8)
#ax=sb.lmplot(x="Ghostcells", y="Overall_Time", data=df5, ci=None, lowess=True)
x=df5["Ghostcells"]
y=df5["Overall_Time"]
new_x, new_y = zip(*sorted(zip(x, y)))

plt.plot(new_x,new_y,color="darkblue")
plt.plot(new_x,new_y,"o",color="darkblue",label="1D Implementierung (32 Prozesse)")

x=df6["Ghostcells"]
y=df6["Overall_Time"]
#new_x, new_y = zip(*sorted(zip(x, y)))

plt.plot(x,y,color="lightblue")
plt.plot(x,y,"o",color="lightblue",label="1D Implementierung (37 Prozesse)")
#ax.set(xlim=(0, 110))
plt.xlabel('Geisterzellenbreite')
plt.ylabel('Zeit Sekunden')
plt.legend()
plt.savefig("Ghostcells_1000.png")

df1=df.query('Num_Iterations==1000 and N == 10000 and Output == False and Ghostcells==10 ')

df2=df.query('Num_Iterations==1000 and N == 10000 and Output == False and Ghostcells==10 and Stepsolver2==True')
df3=df.query('Num_Iterations==1000 and N == 10000 and Output == False and Ghostcells==10 and Stepsolver2==False')



#figure 3

plt.figure(3)


plt.suptitle("Gesamtlaufzeit in Abbhängigkeit von der Anzahl der Prozesse", fontsize=10)
plt.title("N: 10000, α=0.000001, Iterations=1000, Geisterzellenbreite=10 ", fontsize=8)

plt.xlabel("Processes")
plt.ylabel("Overall Time")


x=df2["Processes"]
y=df2["Overall_Time"]
new_x, new_y = zip(*sorted(zip(x, y)))

plt.plot(new_x,new_y,color="darkblue")
plt.plot(df2["Processes"], df2["Overall_Time"], 'o',label="2D Implementierung",color="darkblue")

x=df3["Processes"]
y=df3["Overall_Time"]
new_x, new_y = zip(*sorted(zip(x, y)))

plt.plot(new_x,new_y,color="green")

plt.plot(df3["Processes"], df3["Overall_Time"], 'o',label="1D Implementierung",color="green")

plt.legend()
plt.savefig("Process_Time.png")

import pygame
import math
import pyaudio
import numpy
import time
import wave
import random
from scipy.io.wavfile import read
from PIL import Image
#from numpy.random import seed
#from numpy.random import randint
#rom numpy.random import normal
#import scipy.fftpack
#from functools import lru_cache

from video import make_video
pygame.init()

m = 1
surface_size = 1000*m

main_surface = pygame.display.set_mode((1000,1000))#pygame.SRCALPHA)#,pygame.DOUBLEBUF)#+pygame.FULLSCREEN)
my_clock = pygame.time.Clock()
array = numpy.zeros([int(1000*m),int(1000*m),3], dtype=numpy.uint8)

#lru_cache(maxsize=None)
def draw_tree(inc3, gold, wave, inc, half, inord, order, theta, thetab, sz, posn, heading, color=(0,0,0), depth=0):
   #wave = 0
   trunk_ratio = (1 + 5 ** 0.5) / 2+(1*wave)#(0.2*math.sin(math.pi*inc))
   #trunk_ratio = 1.324717957244746025960908854+wave
   trunk = sz * trunk_ratio
   #nn = 32
   delta_y = trunk * ((math.sin(1*inc3*heading.real*400/400).real))+(math.cos(math.cos(inc3*heading.real*400/400).real))#*((math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1))#*(math.cos(3*(heading*400/400).real)/3)#+math.sin((heading*400/400).real)
   #delta_y = trunk*math.cos((heading*400/400).real)
   delta_x = trunk * ((math.cos(1*inc3*heading.real*400/400).real))+(math.sin(math.sin(inc3*heading.real*400/400).real))#*((math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1))#*(math.cos(3*(heading*400/400).real)/3)#-math.sin((delta_y*400/400).real)#(1 * math.sin((heading*400/400).real))
   #delta_x = trunk * (math.sin(5*(heading*400/400).real)/5)+(math.sin(3*(heading*400/400).real)/3)+math.sin((heading*400/400).real)

   #thetaj = theta*1j*1j
   #thetai = thetab*1j*1j
   
   (u, v) = posn
   newpos = (u + delta_x, v + delta_y)
   
   #if order<2:
   """
   if order==5:
      if half == 1:
         pygame.draw.line(main_surface, color, newpos, newpos, 1)
   if order==3:
   #if True:
      if half == 1:
         pygame.draw.line(main_surface, color, newpos, newpos, 1)
         #pygame.draw.line(main_surface, color, newpos, posn, 1)
         #pygame.draw.circle(main_surface, color, (int(posn[0]),int(posn[1])),int(1*math.fabs(math.sin((heading*400/400).real))),int(1*math.fabs(math.sin((heading*400/400).real))))
         #pygame.draw.circle(main_surface, color, (int(newpos[0]),int(newpos[1])),int(1*math.fabs(math.sin((heading*400/400).real))),int(1*math.fabs(math.sin((heading*400/400).real))))
         #pygame.draw.circle(main_surface, color, (int(posn[0]),int(newpos[1])),1,1)
         #pygame.draw.circle(main_surface, color, (int(newpos[0]),int(posn[1])),1,1)
         
   if order==3:
   #if True:
    #  if half == 1:
   
         pygame.draw.line(main_surface, color, newpos, newpos, 1)
         """
   #gold = 5
   if True:#order%2 == 0:#order==0 or order==2 or order== 4:
   #if True:
   #if True:
      if True:#half == 1:
         if True:#depth == gold:
            #if (math.sin(math.pi*inc*4)) <=0:
            if newpos[0]<1000*m and newpos[1]<1000*m and newpos[0]>0 and newpos[1]>0:
                  pygame.draw.line(main_surface, color, newpos, newpos, 1)
                  
                  pre  = array[int(newpos[0])][int(newpos[1])][0] 
                  pre1 = array[int(newpos[0])][int(newpos[1])][1] 
                  pre2 = array[int(newpos[0])][int(newpos[1])][2]
                  if (color[0])+pre<=255:
                     array[int(newpos[0])][int(newpos[1])][0] = (color[0])+pre
                  if (color[1])+pre1<=255:
                     array[int(newpos[0])][int(newpos[1])][1] = (color[1])+pre1#255
                  if (color[2])+pre2<=255:
                     array[int(newpos[0])][int(newpos[1])][2] = (color[2])+pre2#255
                     #pygame.draw.line(main_surface, color, newpos, newpos, 1)
         #if depth == 3:
          #  pygame.draw.line(main_surface, color, newpos, newpos, 1)
         #halfpos = [0,0]
         #(halfpos[0], halfpos[1]) = (((posn[0]+newpos[0])/2),((posn[1]+newpos[1])/2))
         #pygame.draw.line(main_surface, color, halfpos, halfpos, 1)
      
   #gradd  = 128+(127*math.cos((math.exp(heading.imag)).imag))
   #hite =  math.cos(8*(heading*1j).imag)
                   
   #if hite <= 0.5:
   #wave = wave
   rain = 4
   phase = 0#0.5*math.pi
   gradd  = (128+(127*-(math.sin(wave*math.pi*rain+phase))))#*abs(math.sin(math.pi*inc*4))
   gradd1  = (128+(127*-(math.cos(wave*math.pi*rain+phase))))#*abs(math.sin(math.pi*inc*4))
   gradd2  = (128+(127*(math.sin(wave*math.pi*rain+phase))))#*abs(math.sin(math.pi*inc*4))
   #gradd  = (128+(127*(-math.sin(wave*math.pi*10))))#*abs(math.sin(math.pi*inc*4))
   #gradd1  = (128+(127*(math.cos(wave*math.pi*10))))#*abs(math.sin(math.pi*inc*4))
   #gradd2  = (128+(127*(-math.sin(wave*math.pi*10))))#*abs(math.sin(math.pi*inc*4))
   #wave = wave/10
   #gradd1  = 128+(127*math.sin(math.cos(math.sin(((1 + 5 ** 0.5) / 2)*16*math.pi*(inc))*(2*(1 + 5 ** 0.5)/2))))
   #gradd2 = 128+(127*math.sin(math.cos(math.sin(((1 + 5 ** 0.5) / 2)*16*math.pi*(inc))*(2*(1 + 5 ** 0.5)/2))))
   #else:
      #gradd  = 128+(127*math.cos(8*(-heading*1j).imag))
      #gradd1  = 128+(127*math.cos(8*(-heading*1j).imag))
      #gradd2  = 128+(127*math.cos(8*(-heading*1j).imag))
   

   if order > 0:
      #if depth == 0:
          #color1 = (gradd,gradd1,gradd2)
          #color2 = (gradd,gradd2,gradd3)
          #color1 = (gradd,gradd1,gradd2,255)
          #color2 = (gradd,gradd,gradd,255)
      #else:
         #pass
          #color1 = (gradd,gradd2,255)
          #color2 = (gradd,gradd2,255)
          #color1 = (gradd,gradd2,255)
          #color2 = (gradd,gradd2,255)
      
      color1 = (gradd,gradd1,gradd2)
      
      #color1 = (255,255,255)
      #color2 = (255,255,255)
      # make the recursive calls to draw the two subtrees
      newsz = sz*(1 - trunk_ratio)
      #pygame.display.flip()
      half = 1
      draw_tree(inc3, gold, wave, inc, half, inord, order-1, theta, thetab, newsz, newpos, (heading+theta), color1, depth)
      half = 0
      #draw_tree(gold, wave, inc, half, inord, order-1, theta, thetab, newsz, newpos, (heading+thetab), color1, depth+1)

      
#@lru_cache(maxsize=None)
def draw_tree2(inc3, gold, wave, inc, half, inord, order, theta, thetab, sz, posn, heading, color=(0,0,0), depth=0):
   #wave = 0
   #trunk_ratio = (1 + 5 ** 0.5) / 2#+(1*wave)#(0.2*math.sin(math.pi*inc))
   trunk_ratio = 1.324717957244746025960908854+wave
   trunk = sz * trunk_ratio
   #nn = 32
   delta_x = trunk * ((math.sin(8*inc3*heading.real*400/400).real))*((math.cos(inc3*heading.real*400/400).real))#*((math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1))#*(math.cos(3*(heading*400/400).real)/3)#+math.sin((heading*400/400).real)
   #delta_y = trunk*math.cos((heading*400/400).real)
   delta_y = trunk * ((math.cos(8*inc3*heading.real*400/400).real))*((math.sin(inc3*heading.real*400/400).real))#*((math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1)*(math.cos(nn*(heading*400/400).real)/1))#*(math.cos(3*(heading*400/400).real)/3)#-math.sin((delta_y*400/400).real)#(1 * math.sin((heading*400/400).real))
   #delta_x = trunk * (math.sin(5*(heading*400/400).real)/5)+(math.sin(3*(heading*400/400).real)/3)+math.sin((heading*400/400).real)

   #thetaj = theta*1j*1j
   #thetai = thetab*1j*1j
   
   (u, v) = posn
   newpos = (u + delta_x, v + delta_y)
   
   #if order<2:
   """
   if order==5:
      if half == 1:
         pygame.draw.line(main_surface, color, newpos, newpos, 1)
   if order==3:
   #if True:
      if half == 1:
         pygame.draw.line(main_surface, color, newpos, newpos, 1)
         #pygame.draw.line(main_surface, color, newpos, posn, 1)
         #pygame.draw.circle(main_surface, color, (int(posn[0]),int(posn[1])),int(1*math.fabs(math.sin((heading*400/400).real))),int(1*math.fabs(math.sin((heading*400/400).real))))
         #pygame.draw.circle(main_surface, color, (int(newpos[0]),int(newpos[1])),int(1*math.fabs(math.sin((heading*400/400).real))),int(1*math.fabs(math.sin((heading*400/400).real))))
         #pygame.draw.circle(main_surface, color, (int(posn[0]),int(newpos[1])),1,1)
         #pygame.draw.circle(main_surface, color, (int(newpos[0]),int(posn[1])),1,1)
         
   if order==3:
   #if True:
    #  if half == 1:
   
         pygame.draw.line(main_surface, color, newpos, newpos, 1)
         """
   #gold = 5
   if True:#order%2 == 0:#order==0 or order==2 or order== 4:
   #if True:
   #if True:
      if True:#half == 1:
         if True:#depth == gold:
            #if (math.sin(math.pi*inc*4)) <=0:
            if newpos[0]<1000*m and newpos[1]<1000*m and newpos[0]>0 and newpos[1]>0:
                  pygame.draw.line(main_surface, color, newpos, newpos, 1)
                  
                  pre  = array[int(newpos[0])][int(newpos[1])][0] 
                  pre1 = array[int(newpos[0])][int(newpos[1])][1] 
                  pre2 = array[int(newpos[0])][int(newpos[1])][2]
                  if (color[0])+pre<=255:
                     array[int(newpos[0])][int(newpos[1])][0] = (color[0])+pre
                  if (color[1])+pre1<=255:
                     array[int(newpos[0])][int(newpos[1])][1] = (color[1])+pre1#255
                  if (color[2])+pre2<=255:
                     array[int(newpos[0])][int(newpos[1])][2] = (color[2])+pre2#255
                     #pygame.draw.line(main_surface, color, newpos, newpos, 1)
         #if depth == 3:
          #  pygame.draw.line(main_surface, color, newpos, newpos, 1)
         #halfpos = [0,0]
         #(halfpos[0], halfpos[1]) = (((posn[0]+newpos[0])/2),((posn[1]+newpos[1])/2))
         #pygame.draw.line(main_surface, color, halfpos, halfpos, 1)
      
   #gradd  = 128+(127*math.cos((math.exp(heading.imag)).imag))
   #hite =  math.cos(8*(heading*1j).imag)
                   
   #if hite <= 0.5:
   #wave = wave
   
   gradd  = (128+(127*-(math.sin(wave*math.pi*10))))#*abs(math.sin(math.pi*inc*4))
   gradd1  = (128+(127*-(math.cos(wave*math.pi*10))))#*abs(math.sin(math.pi*inc*4))
   gradd2  = (128+(127*(math.sin(wave*math.pi*10))))#*abs(math.sin(math.pi*inc*4))
   #gradd  = (128+(127*(-math.sin(wave*math.pi*10))))#*abs(math.sin(math.pi*inc*4))
   #gradd1  = (128+(127*(math.cos(wave*math.pi*10))))#*abs(math.sin(math.pi*inc*4))
   #gradd2  = (128+(127*(-math.sin(wave*math.pi*10))))#*abs(math.sin(math.pi*inc*4))
   #wave = wave/10
   #gradd1  = 128+(127*math.sin(math.cos(math.sin(((1 + 5 ** 0.5) / 2)*16*math.pi*(inc))*(2*(1 + 5 ** 0.5)/2))))
   #gradd2 = 128+(127*math.sin(math.cos(math.sin(((1 + 5 ** 0.5) / 2)*16*math.pi*(inc))*(2*(1 + 5 ** 0.5)/2))))
   #else:
      #gradd  = 128+(127*math.cos(8*(-heading*1j).imag))
      #gradd1  = 128+(127*math.cos(8*(-heading*1j).imag))
      #gradd2  = 128+(127*math.cos(8*(-heading*1j).imag))
   

   if order > 0:
      #if depth == 0:
          #color1 = (gradd,gradd1,gradd2)
          #color2 = (gradd,gradd2,gradd3)
          #color1 = (gradd,gradd1,gradd2,255)
          #color2 = (gradd,gradd,gradd,255)
      #else:
         #pass
          #color1 = (gradd,gradd2,255)
          #color2 = (gradd,gradd2,255)
          #color1 = (gradd,gradd2,255)
          #color2 = (gradd,gradd2,255)
      
      color1 = (gradd,gradd1,gradd2)
      
      #color1 = (255,255,255)
      #color2 = (255,255,255)
      # make the recursive calls to draw the two subtrees
      newsz = sz*(1 - trunk_ratio)
      #pygame.display.flip()
      half = 1
      draw_tree2(inc3, gold, wave, inc, half, inord, order-1, theta, thetab, newsz, newpos, (heading+theta), color1, depth)
      half = 0
      #draw_tree(gold, wave, inc, half, inord, order-1, theta, thetab, newsz, newpos, (heading+thetab), color1, depth+1)

      
def draw_tree3(inord, order, theta, thetab, sz, posn, heading, color=(0,0,0), depth=0):

   trunk_ratio = (1 + 5 ** 0.5) / 2       
   trunk = sz * trunk_ratio
   delta_x = trunk * -math.cos((heading*400/400).real)
   delta_y = trunk * -math.sin((heading*400/400).real)

   thetaj = theta*1j*1j
   thetai = thetab*1j*1j
   
   (u, v) = posn
   newpos = (u + delta_x, v + delta_y)
   #if order<2:
   #if order==1:
   if True:
      #pygame.draw.line(main_surface, color, posn, newpos, order)
      #pygame.draw.line(main_surface, color, newpos, posn, order)
      pygame.draw.circle(main_surface, color, (int(posn[0]),int(posn[1])),int(1*math.fabs(math.sin((heading*400/400).real))),int(1*math.fabs(math.sin((heading*400/400).real))))
      pygame.draw.circle(main_surface, color, (int(newpos[0]),int(newpos[1])),int(1*math.fabs(math.sin((heading*400/400).real))),int(1*math.fabs(math.sin((heading*400/400).real))))
      #pygame.draw.circle(main_surface, color, (int(posn[0]),int(newpos[1])),1,1)
      #pygame.draw.circle(main_surface, color, (int(newpos[0]),int(posn[1])),1,1)
      
   #gradd  = 128+(127*math.cos((math.exp(heading.imag)).imag))
   gradd  = 128+(127*math.cos((heading*1).imag))
   gradd2 = 128+(127*math.sin((heading*1j*1).real))
   gradd3 = 128+(128*math.sin((heading).real))
   

   if order > 0:
      if depth == 0:
          color1 = (255,gradd2,gradd)
          color2 = (0,gradd2,gradd)
      else:
          color1 = (gradd2,255,gradd)
          color2 = (gradd2,0,gradd)#(gradd,gradd2,0)


      # make the recursive calls to draw the two subtrees
      newsz = sz*(1 - trunk_ratio)
      #pygame.display.flip()
      draw_tree3(inord, order-1, theta, thetab, newsz, newpos, (heading+theta+12j), color1, depth)
      draw_tree3(inord, order-1, theta, thetab, newsz, newpos, (heading+thetab+12j), color2, depth)

def draw_tree4(inord, order, theta, thetab, sz, posn, heading, color=(0,0,0), depth=0):

   trunk_ratio = (1 + 5 ** 0.5) / 2       
   trunk = sz * trunk_ratio
   delta_x = trunk * math.cos((heading*400/400).real)
   delta_y = trunk * math.sin((heading*400/400).real)

   thetaj = theta*1j*1j
   thetai = thetab*1j*1j
   
   (u, v) = posn
   newpos = (u + delta_x, v + delta_y)
   #if order<2:
   #if order==1:
   if True:
      #pygame.draw.line(main_surface, color, posn, newpos, order)
      #pygame.draw.line(main_surface, color, newpos, posn, order)
      pygame.draw.circle(main_surface, color, (int(posn[0]),int(posn[1])),int(1*math.fabs(math.sin((heading*400/400).real))),int(1*math.fabs(math.sin((heading*400/400).real))))
      pygame.draw.circle(main_surface, color, (int(newpos[0]),int(newpos[1])),int(1*math.fabs(math.sin((heading*400/400).real))),int(1*math.fabs(math.sin((heading*400/400).real))))
      #pygame.draw.circle(main_surface, color, (int(posn[0]),int(newpos[1])),1,1)
      #pygame.draw.circle(main_surface, color, (int(newpos[0]),int(posn[1])),1,1)
      
   #gradd  = 128+(127*math.cos((math.exp(heading.imag)).imag))
   gradd  = 128+(127*math.cos((heading*1).imag))
   gradd2 = 128+(127*math.sin((heading*1j*1).real))
   gradd3 = 128+(128*math.sin((heading).real))
   

   if order > 0:
      if depth == 0:
          color1 = (255,gradd,gradd2)
          color2 = (0,gradd2,gradd2)
      else:
          color1 = (gradd2,255,gradd)
          color2 = (gradd2,0,gradd)#(gradd,gradd2,0)


      # make the recursive calls to draw the two subtrees
      newsz = sz*(1 - trunk_ratio)
      #pygame.display.flip()
      draw_tree4(inord, order-1, theta, thetab, newsz, newpos, (heading+theta+12j), color1, depth)
      draw_tree4(inord, order-1, theta, thetab, newsz, newpos, (heading+thetab+12j), color2, depth)
#@lru_cache(maxsize=None)
def gameloop():
    fileno = 0
    theta1 = 0
    theta2 = 0
    frame = 0
    inord = 0
    headingin = 0
    half = 0
    WIDTH = 4
    CHANNELS = 1
    RATE = 22050
    incre = 0
    incre2 = 0
    incre3 = 1#(1/100)*346#1#2400/1800
    gold = 1
    #incre2 = -1
##    a = read("sweep.wav")
##    b = read("neg.wav")
##    songarr = numpy.array(a[1],dtype=float)
##    songarr2 = numpy.array(b[1],dtype=float)
    save_screen = make_video(main_surface, "test1")

    #main_surface.fill((255,255,255))
    main_surface.fill((0,0,0))
    
   
    def drawer(wavey, gold, inc, ran, ran2, ra, ra2, inc3):
       #pygame.display.flip()
       #my_clock.tick(60)
       #next(save_screen)
        
        wavery = (numpy.random.normal(loc=0.0, scale=0.03, size=None))#((wavey/(18761*10)))
        woo = ((wavey/31452))
        woo2 = math.exp(2+abs(woo))
        inc = inc #- woo2/25
        randoo= (numpy.random.normal(loc=0.0, scale=1, size=None))#(math.cos(random.random()*1000*inc*math.pi))
        theta1a = ((math.sin(inc3*math.pi*(inc)))*1)*math.pi#*randoo#(1 + 5 ** 0.5))#*((math.cos(200*math.pi*(inc)))*math.pi)#*random.random()#*(gold*(1 + 5 ** 0.5)/2*(ra))#+(4*math.cos(math.sin(1*math.pi*(inc))*(2*math.pi*(ran))))**(1*math.cos(math.cos(2*math.pi*(inc))*(2*math.pi*(ran)))))#+math.cos(4*math.pi*(inc))*(2*math.pi*(ran))**math.cos(math.cos(4*math.pi*(inc))*(2*math.pi*(ran)))
        theta1 = theta1a#(math.sin(2*(inc)/2))+(math.sin(4*(inc)/4))+(math.sin(6*(inc)/6))+(math.sin(8*(inc)/8))+(math.sin(10*(inc)/10))+(math.sin(12*(inc)/12))
        theta11a = ((math.cos(inc3*math.pi*(inc)))*1)*math.pi#*randoo#(1 + 5 ** 0.5))#*((math.cos(200*math.pi*(inc)))*math.pi)#*random.random()#*(gold*(1 + 5 ** 0.5)/2*(ra2))#+(4*math.cos(math.sin(1*math.pi*(inc))*(2*math.pi*(ran2))))**(1*math.cos(math.cos(2*math.pi*(inc))*(2*math.pi*(ran)))))#+math.cos(4*math.pi*(inc))*(2*math.pi*(ran2))**math.cos(math.cos(4*math.pi*(inc))*(2*math.pi*(ran2)))
        theta11 = theta11a#+(math.sin(2*(inc)/2))+(math.sin(4*(inc)/4))+(math.sin(6*(inc)/6))+(math.sin(8*(inc)/8))+(math.sin(10*(inc)/10))+(math.sin(12*(inc)/12))
        #theta1 = math.sin(4*math.pi*(inc))*(2*math.pi*(ran))+math.sin(8*math.pi*(inc))*(2*math.pi*(ran))+math.sin(2*math.pi*(inc))*(2*math.pi*(ran))
        #theta11 = math.sin(4*math.pi*(inc))*(2*math.pi*(ran2))+math.sin(8*math.pi*(inc))*(2*math.pi*(ran2))+math.sin(2*math.pi*(inc))*(2*math.pi*(ran2))
        
        theta2a =math.pi#+((math.cos(1*math.pi*(inc)))*1)*math.pi # (math.sin(200*inc*math.pi))#(math.sin(4*math.pi*(inc)))*math.pi#*(gold*(1 + 5 ** 0.5)/2*(ra))#abs(math.sin(50*math.pi*(inc)))#+math.sin(math.pi*1*random.random())#((1 + 5 ** 0.5)/2*(ra)))
        theta2 = math.pi#+((math.sin(1*math.pi*(inc)))*1)*math.pi#0#.5*math.pi#theta2a+(math.sin(4*(inc)/4))+(math.sin(6*(inc)/6))+(math.sin(8*(inc)/8))+(math.sin(10*(inc)/10))+(math.sin(12*(inc)/12))+(math.sin(14*(inc)/14))
        theta22a = 1#abs((math.cos(0.125*math.pi*(inc)))*1)#(math.sin(4*math.pi*(inc)))*math.pi#+math.sin(math.pi*1*random.random())#*((1 + 5 ** 0.5)/2*(ra2)))
        theta22 = 1#abs((math.cos(0.125*math.pi*(inc)))*1)#theta22a+(math.sin(4*(inc)/4))+(math.sin(6*(inc)/6))+(math.sin(8*(inc)/8))+(math.sin(10*(inc)/10))+(math.sin(12*(inc)/12))+(math.sin(14*(inc)/14))
        #wavery = 0
        iterr = 2
        #if math.sin(math.pi*inc)>=0:
        draw_tree(-wavery, inc, half, inord, iterr, (-theta1), (theta2a), surface_size*0.15*ran*theta22a, (500*m,500*m), (1*math.pi*inc))
        draw_tree(-wavery, inc, half, inord, iterr, (-theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*math.pi*inc))
        draw_tree(wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran*theta22a, (500*m,500*m), (1*math.pi*inc))
        draw_tree(wavery, inc, half, inord, iterr, (theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*math.pi*inc))
        
        #draw_tree(-wavery, inc, half, inord, iterr, (-theta1), (theta2a), surface_size*0.15*ran*theta22a-woo, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(-wavery, inc, half, inord, iterr, (-theta11), (theta2a), surface_size*0.15*ran2*-theta22-woo, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(-wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran*theta22a-woo, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(-wavery, inc, half, inord, iterr, (theta11), (theta2a), surface_size*0.15*ran2*-theta22-woo, (500*m,500*m), (1*math.pi*inc))
        """
        draw_tree(wavery, inc, half, inord, iterr, (-theta1), (theta2), surface_size*0.15*ran*theta22, (500*m,500*m), (1*math.pi*inc))
        draw_tree(wavery, inc, half, inord, iterr, (-theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*-math.pi*inc))
        draw_tree(wavery, inc, half, inord, iterr, (theta1), (theta2), surface_size*0.15*ran*theta22, (500*m,500*m), (1*math.pi*inc))
        draw_tree(wavery, inc, half, inord, iterr, (theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*-math.pi*inc))
        
        draw_tree(-wavery, inc, half, inord, iterr, (-theta1), (theta2a), surface_size*0.15*ran*theta22a, (500*m,500*m), (1*math.pi*inc))
        draw_tree(-wavery, inc, half, inord, iterr, (-theta11), (theta2a), surface_size*0.15*ran2*-theta22a, (500*m,500*m), (1*-math.pi*inc))
        draw_tree(-wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran*theta22a, (500*m,500*m), (1*math.pi*inc))
        draw_tree(-wavery, inc, half, inord, iterr, (theta11), (theta2a), surface_size*0.15*ran2*-theta22a, (500*m,500*m), (1*-math.pi*inc))
        
        draw_tree(-wavery, inc, half, inord, iterr, (-theta1), (theta2), surface_size*0.15*ran*theta22, (500*m,500*m), (1*math.pi*inc))
        draw_tree(-wavery, inc, half, inord, iterr, (-theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*-math.pi*inc))
        draw_tree(-wavery, inc, half, inord, iterr, (theta1), (theta2), surface_size*0.15*ran*theta22, (500*m,500*m), (1*math.pi*inc))
        draw_tree(-wavery, inc, half, inord, iterr, (theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*-math.pi*inc))
        
        """
        #draw_tree(inc, half, inord, 2, (theta1), (theta2), surface_size*0.15*ran, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(inc, half, inord,2, (theta11), (theta22), surface_size*0.15*ran2, (500*m,500*m), (-1*math.pi*inc))
        #draw_tree(half, inord, 5, (theta1), (theta2), surface_size*0.1*ran, (500*m,500*m), (math.pi*inc))
        #draw_tree(half, inord, 5, (theta11), (theta22), surface_size*0.1*ran2, (500*m,500*m), (math.pi*inc))
        #draw_tree(half, inord, 8, (theta1), (theta2), surface_size*0.1*ran, (500*m,500*m), (math.pi*inc))
        #draw_tree(half, inord, 8, (theta11), (theta22), surface_size*0.1*ran2, (500*m,500*m), (math.pi*inc))

    def drawer2(wavey2, wavey, gold, inc, ran, ran2, ra, ra2, inc3):
       #pygame.display.flip()
       #my_clock.tick(60)
       #next(save_screen)

        wavery = ((wavey/(18761*8)))#*(numpy.random.normal(loc=0.0, scale=0.3, size=None))
        wavery2 = ((wavey2/(18761*10)))#*(numpy.random.normal(loc=0.0, scale=0.3, size=None))
        woo = ((wavey/31452))
        woo2 = math.exp(2+abs(woo))
        inc = inc+ numpy.random.randint(0,20000)#(numpy.random.normal(loc=0.0, scale=1, size=None)) #- woo2/25
        incre3 = 1
        randoo= 1#(numpy.random.normal(loc=0.0, scale=1, size=None))#(math.cos(random.random()*1000*inc*math.pi))
        theta1a = (inc*math.pi)*1*math.pi#((math.sin(2*incre3*math.pi*(inc)))*1)*math.pi*randoo#(1 + 5 ** 0.5))#*((math.cos(200*math.pi*(inc)))*math.pi)#*random.random()#*(gold*(1 + 5 ** 0.5)/2*(ra))#+(4*math.cos(math.sin(1*math.pi*(inc))*(2*math.pi*(ran))))**(1*math.cos(math.cos(2*math.pi*(inc))*(2*math.pi*(ran)))))#+math.cos(4*math.pi*(inc))*(2*math.pi*(ran))**math.cos(math.cos(4*math.pi*(inc))*(2*math.pi*(ran)))
        theta1 = theta1a#(math.sin(2*(inc)/2))+(math.sin(4*(inc)/4))+(math.sin(6*(inc)/6))+(math.sin(8*(inc)/8))+(math.sin(10*(inc)/10))+(math.sin(12*(inc)/12))
        theta11a = math.pi#math.cos(4*inc*math.pi)#((math.cos(2*incre3*math.pi*(inc)))*1)*math.pi*randoo#(1 + 5 ** 0.5))#*((math.cos(200*math.pi*(inc)))*math.pi)#*random.random()#*(gold*(1 + 5 ** 0.5)/2*(ra2))#+(4*math.cos(math.sin(1*math.pi*(inc))*(2*math.pi*(ran2))))**(1*math.cos(math.cos(2*math.pi*(inc))*(2*math.pi*(ran)))))#+math.cos(4*math.pi*(inc))*(2*math.pi*(ran2))**math.cos(math.cos(4*math.pi*(inc))*(2*math.pi*(ran2)))
        theta11 = theta11a#+(math.sin(2*(inc)/2))+(math.sin(4*(inc)/4))+(math.sin(6*(inc)/6))+(math.sin(8*(inc)/8))+(math.sin(10*(inc)/10))+(math.sin(12*(inc)/12))
        #theta1 = math.sin(4*math.pi*(inc))*(2*math.pi*(ran))+math.sin(8*math.pi*(inc))*(2*math.pi*(ran))+math.sin(2*math.pi*(inc))*(2*math.pi*(ran))
        #theta11 = math.sin(4*math.pi*(inc))*(2*math.pi*(ran2))+math.sin(8*math.pi*(inc))*(2*math.pi*(ran2))+math.sin(2*math.pi*(inc))*(2*math.pi*(ran2))
        theta2a =0#math.pi#(inc*math.pi)*2*math.pi#math.pi#math.sin(-inc*math.pi)*2*math.pi*incre3#math.cos(incre3*inc*math.pi)*2*math.pi*incre3#incre3*math.pi*inc#(1 + 5 ** 0.5)#math.pi#+((math.cos(1*math.pi*(inc)))*1)*math.pi # (math.sin(200*inc*math.pi))#(math.sin(4*math.pi*(inc)))*math.pi#*(gold*(1 + 5 ** 0.5)/2*(ra))#abs(math.sin(50*math.pi*(inc)))#+math.sin(math.pi*1*random.random())#((1 + 5 ** 0.5)/2*(ra)))
        theta2 = 0#math.pi#(1 + 5 ** 0.5)#math.pi#+((math.sin(1*math.pi*(inc)))*1)*math.pi#0#.5*math.pi#theta2a+(math.sin(4*(inc)/4))+(math.sin(6*(inc)/6))+(math.sin(8*(inc)/8))+(math.sin(10*(inc)/10))+(math.sin(12*(inc)/12))+(math.sin(14*(inc)/14))
        theta22a = 1#abs((math.cos(0.125*math.pi*(inc)))*1)#(math.sin(4*math.pi*(inc)))*math.pi#+math.sin(math.pi*1*random.random())#*((1 + 5 ** 0.5)/2*(ra2)))
        theta22 = math.pi#abs((math.cos(0.125*math.pi*(inc)))*1)#theta22a+(math.sin(4*(inc)/4))+(math.sin(6*(inc)/6))+(math.sin(8*(inc)/8))+(math.sin(10*(inc)/10))+(math.sin(12*(inc)/12))+(math.sin(14*(inc)/14))
        #wavery = 0
        itermod = abs(math.cos(0.5*incre3*inc*math.pi))
        if itermod <= 0.25:
           iterr = 5
        elif itermod <= 0.5:
           iterr = 0
        elif itermod <= 0.75:
           iterr = 0
        elif itermod <= 1:
           iterr = 2
        iterr = 8
        #if math.sin(math.pi*inc)>=0:
        inc = (inc)
        ran2= 0.1#.297872340425532
        mins = -1*math.pi
        tet = 0
        for s in range(12):
           tet = (s+1)*2
           for xx in range(tet):
              re = -(0.125/5)+(numpy.random.randint(0,200000)/500000)
              re2 = re*1000
              wavery2 = wavery +re#abs(numpy.random.normal(loc=0.0, scale=0.1, size=None))#1*math.sin(100*inc*math.pi)
              draw_tree(inc3, gold, wavery2, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a*tet, (500*m,500*m), (mins+(xx*(1/((tet)/2))*math.pi)))
           #draw_tree2(inc3, gold, wavery2, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (mins+(xx*0.25*math.pi)))
           #draw_tree(gold, wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (mins+(x*0.25*math.pi)))
        
##        wavery = wavery + abs(numpy.random.randint(0,20000)/100000)#abs(numpy.random.normal(loc=0.0, scale=0.1, size=None))#1*math.sin(100*inc*math.pi)
##        draw_tree(gold, wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi))
##        wavery = wavery + abs(numpy.random.randint(0,20000)/100000)
##        draw_tree(gold, wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (0))
##        wavery = wavery + abs(numpy.random.randint(0,20000)/100000)
##        draw_tree(gold, wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*0.5))
##        wavery = wavery + abs(numpy.random.randint(0,20000)/100000)
##        draw_tree(gold, wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*-0.5))
##        wavery = wavery + abs(numpy.random.randint(0,20000)/100000)
##        draw_tree(gold, wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*0.25))
##        wavery = wavery + abs(numpy.random.randint(0,20000)/100000)
##        draw_tree(gold, wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*0.75))
##        wavery = wavery + abs(numpy.random.randint(0,20000)/100000)
##        draw_tree(gold, wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*-0.25))
##        wavery = wavery + abs(numpy.random.randint(0,20000)/100000)
##        draw_tree(gold, wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*-0.75))
        #draw_tree(gold, wavery, inc, half, inord, iterr, (-theta1), (-theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (-1*math.pi*inc))
        #draw_tree(gold, wavery, inc, half, inord, iterr, (-theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*math.pi*inc))
##        draw_tree(gold, wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*inc))
##        #draw_tree(gold, wavery, inc, half, inord, iterr, (theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*math.pi*inc))
##
##        draw_tree(gold, -wavery, inc, half, inord, iterr, (-theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*inc))
##        #draw_tree(gold, -wavery, inc, half, inord, iterr, (-theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*math.pi*inc))
##        draw_tree(gold, -wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*inc))
##        #draw_tree(gold, -wavery, inc, half, inord, iterr, (theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*math.pi*inc))
##
        #inc = -inc#(numpy.random.normal(loc=0.0, scale=0.01, size=None))
        #draw_tree(-(numpy.random.normal(loc=0.0, scale=0.03, size=None)), inc, half, inord, iterr, (-theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(-(numpy.random.normal(loc=0.0, scale=0.03, size=None)), inc, half, inord, iterr, (-theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*math.pi*inc))
        #draw_tree((numpy.random.normal(loc=0.0, scale=0.03, size=None)), inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*inc))
        #draw_tree((numpy.random.normal(loc=0.0, scale=0.03, size=None)), inc, half, inord, iterr, (theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*math.pi*inc))
        
        #draw_tree(-0, inc, half, inord, iterr, (-theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(-0, inc, half, inord, iterr, (-theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(0, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(0, inc, half, inord, iterr, (theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*math.pi*inc))
        #inc = -inc
        #draw_tree(-0, inc, half, inord, iterr, (-theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(-0, inc, half, inord, iterr, (-theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(0, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran2*theta22a, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(0, inc, half, inord, iterr, (theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(-wavery, inc, half, inord, iterr, (-theta1), (theta2a), surface_size*0.15*ran*theta22a-woo, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(-wavery, inc, half, inord, iterr, (-theta11), (theta2a), surface_size*0.15*ran2*-theta22-woo, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(-wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran*theta22a-woo, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(-wavery, inc, half, inord, iterr, (theta11), (theta2a), surface_size*0.15*ran2*-theta22-woo, (500*m,500*m), (1*math.pi*inc))
        """
        draw_tree(wavery, inc, half, inord, iterr, (-theta1), (theta2), surface_size*0.15*ran*theta22, (500*m,500*m), (1*math.pi*inc))
        draw_tree(wavery, inc, half, inord, iterr, (-theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*-math.pi*inc))
        draw_tree(wavery, inc, half, inord, iterr, (theta1), (theta2), surface_size*0.15*ran*theta22, (500*m,500*m), (1*math.pi*inc))
        draw_tree(wavery, inc, half, inord, iterr, (theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*-math.pi*inc))
        
        draw_tree(-wavery, inc, half, inord, iterr, (-theta1), (theta2a), surface_size*0.15*ran*theta22a, (500*m,500*m), (1*math.pi*inc))
        draw_tree(-wavery, inc, half, inord, iterr, (-theta11), (theta2a), surface_size*0.15*ran2*-theta22a, (500*m,500*m), (1*-math.pi*inc))
        draw_tree(-wavery, inc, half, inord, iterr, (theta1), (theta2a), surface_size*0.15*ran*theta22a, (500*m,500*m), (1*math.pi*inc))
        draw_tree(-wavery, inc, half, inord, iterr, (theta11), (theta2a), surface_size*0.15*ran2*-theta22a, (500*m,500*m), (1*-math.pi*inc))
        
        draw_tree(-wavery, inc, half, inord, iterr, (-theta1), (theta2), surface_size*0.15*ran*theta22, (500*m,500*m), (1*math.pi*inc))
        draw_tree(-wavery, inc, half, inord, iterr, (-theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*-math.pi*inc))
        draw_tree(-wavery, inc, half, inord, iterr, (theta1), (theta2), surface_size*0.15*ran*theta22, (500*m,500*m), (1*math.pi*inc))
        draw_tree(-wavery, inc, half, inord, iterr, (theta11), (theta2), surface_size*0.15*ran2*-theta22, (500*m,500*m), (1*-math.pi*inc))
        
        """
        #draw_tree(inc, half, inord, 2, (theta1), (theta2), surface_size*0.15*ran, (500*m,500*m), (1*math.pi*inc))
        #draw_tree(inc, half, inord,2, (theta11), (theta22), surface_size*0.15*ran2, (500*m,500*m), (-1*math.pi*inc))
        #draw_tree(half, inord, 5, (theta1), (theta2), surface_size*0.1*ran, (500*m,500*m), (math.pi*inc))
        #draw_tree(half, inord, 5, (theta11), (theta22), surface_size*0.1*ran2, (500*m,500*m), (math.pi*inc))
        #draw_tree(half, inord, 8, (theta1), (theta2), surface_size*0.1*ran, (500*m,500*m), (math.pi*inc))
        #draw_tree(half, inord, 8, (theta11), (theta22), surface_size*0.1*ran2, (500*m,500*m), (math.pi*inc))
        

    while True:
        #pygame.display.flip()
        # Handle evente from keyboard, mouse, etc.
        #incre3+= 1#math.tan(math.pi*incre)
        
        if incre<3:#*math.pi:
         pass
        else:
          # pygame.display.flip()
           #wavet = songarr[round(incre3)]
           #print("hello")
           #Image.fromarray(array).convert("RGB").save("stripe"+str(fileno)+".png")
           im = Image.fromarray(array).convert("RGB")
           im.save("zzzzzzzzzzz"+str(fileno)+".png")
           im.thumbnail((1800,1800))
           im.save("zzzzzzzzzzzsz"+str(fileno)+".png")
           array.fill(0)
           fileno +=1
           #next(save_screen)
           gold -= 1
           incre = 0
           #incre2+= 1/1800
           incre3 += 1/600
           #incre3=2400/1800
           if gold==0:
              #next(save_screen)
              main_surface.fill((0,0,0))
              gold = 1
              #incre2 += 1/(2/(2/((len(songarr)/30)/60)/100))
              #pygame.quit()
        #print(2/((len(songarr)/30)/30))
            
        incre += 1/100000#/((len(songarr)/30)/60)/100#(1 + 5 ** 0.5) / 2000
        incre2+= 0.0000001#math.tan(math.pi*incre)

        wavey = 1#songarr[round(incre2)]
        wavey2 = 1#songarr2[round(incre2)]
        rand = 0.6#*math.sin(32*math.pi*incre)
        rand2 = 0.6#*math.sin(32*math.pi*incre)
        rad = 1#incre#*(math.sin(4*math.pi*incre))#(math.sin(1/8*incre))
        rad2 = 1#*(math.sin(4*math.pi*incre))#(math.sin(1/8*incre))
        #drawer2(wavey, gold, incre, rand, 1*rand2, rad, rad2, incre3)
        drawer2(wavey2, wavey, gold, incre, rand, 1*rand2, rad, rad2, incre3)
        #drawer2(wavey, gold, incre, rand, 1*rand2, rad, rad2, incre3/4)
        #drawer2(wavey, gold, incre, rand, 10*rand2/36, rad, rad2, incre3/6)
        ev = pygame.event.poll()
        if ev.type == pygame.QUIT:
            break;
        elif ev.type == pygame.KEYDOWN  and ev.key == pygame.K_s:
            im = Image.fromarray(array).convert("RGB")
            im.save("zzzzzzzzzzz.png")
            im.thumbnail((1800,1800))
            im.save("zzzzzzzzzzzsz.png")
            
            #next(save_screen)
            #pygame.display.flip()
        # Updates - change the angle
        #theta1 += 0.002
        #theta2 -= 0.002
        
            
        # Draw everything
        #main_surface.fill((0, 0, 0))
        #draw_tree(8, theta1, theta2, surface_size, (45

gameloop()
pygame.quit()

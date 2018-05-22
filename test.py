#! /usr/bin/env python3
#! /home/psacawa/anaconda3/bin/python

import multiprocessing.dummy as multithread
import time
import keyboard

def _find_getch():
    try:
        import termios
    except ImportError:
        # Non-POSIX. Return msvcrt's (Windows') getch.
        import msvcrt
        return msvcrt.getch

    # POSIX system. Create and return a getch that manipulates the tty.
    import sys, tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    return _getch
#  getch = _find_getch()
stop = False

def keyHandler ():
    stop = False
    while True:
        #  key = getch ()
        #  if key == "q":
        print (stop)
        if keyboard.is_pressed ('q'):
            print ("Pressed q")
            stop = True

def main ():
    p = multithread.Process (target = keyHandler)
    p.start ()
    while not stop:
        #  print (stop)
        print ("do computation")
        time.sleep (1.0)
    pass
    p.join ()

if __name__ == "__main__":
    main ()

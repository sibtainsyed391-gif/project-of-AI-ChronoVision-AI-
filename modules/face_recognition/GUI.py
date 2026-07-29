import tkinter as tk
import time
import threading as th
import _thread

class Employee(th.Thread):

    #root = tk.Tk()


    def __init__(self, root, name_of_employee):
        th.Thread.__init__(self)
        self.start()

        self.name = name_of_employee
        self.is_present = False
        self.Main_frame = tk.Frame(root,bg='#FF0000')
        self.Main_frame.pack(side=tk.TOP)
        self.L_Name = tk.Label(self.Main_frame,text=name_of_employee,bg='#FF0000',width = 30)
        self.L_Name.pack(side=tk.LEFT)
        self.L_time = tk.Label(self.Main_frame,text='Not present yet',bg='#FF0000',width = 20)
        self.L_time.pack(side=tk.LEFT)
        root.update()


    def make_present(self,time):
        self.Time_stamp = str(time)

        self.L_time.configure(bg='#00FF00',text = self.Time_stamp)
        self.L_Name.config(bg='#00FF00')

    def make_absent(self):
        self.L_time.configure(bg='#FF0000', text='Not Present Yet')
        self.L_Name.config(bg='#FF0000')




if __name__ == '__main__':
    root = tk.Tk()

    E1 = Employee(root,'Sanchit')
    root.mainloop()
    E1.make_present(time.ctime())


import tkinter as tk

windowSize = "500x600"

anzahlBalken = 10

class Application(tk.LabelFrame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(side="bottom")
        self.create_widgets()

    def create_widgets(self):
        # self.hi_there = tk.Button(self)
        # self.hi_there["text"] = "Hello World\n(click me)"
        # self.hi_there["command"] = self.say_hi
        # self.hi_there.pack(side="top")

        self.sort = tk.Button(self, text="SORT", fg="blue")
        self.sort['command'] = self.startSort
        self.sort.pack(side="bottom")

        self.create = tk.Button(self, text="CREATE BARS", fg="green")
        self.create['command'] = self.createBars
        self.create.pack(side="bottom")

        self.barsNumber = tk.Entry(self, text="entry")
        self.barsNumber.pack(side="left")

    def say_hi(self):
        print("hi there, everyone!")

    def startSort(self):
        print('sort')

    def createBars(self):
        #value = int(self.barsNumber.get())
        #print(value)
        return 10


class Bar(tk.Button):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        #self.pack=(side="top")
        self.text = "asd"


keinEingabe = True

root = tk.Tk()
root.geometry(windowSize)
window = Application(master=root)

while keinEingabe:
    print('as')
    number = window.createBars()
    if number == 0:
        print('nix')
    else:
        for x in range(number):
            bar = Bar(root)
        keinEingabe = False
window.mainloop()



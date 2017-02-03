import tkinter
from PIL import ImageTk, Image
import glob
import shutil
import sys
import os


classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


class LabelHelper(tkinter.Frame):
    def __init__(self, parent):
        tkinter.Frame.__init__(self, parent)
        self.parent = parent
        self.idx = 0
        self.images = []
        self.workdir = None

    def init(self, path, classname, idx):
        self.workdir = path
        self.classname = classname
        self.idx = idx
        print('Init: ', path, classname, idx)

        if os.path.isdir(self.workdir):
            print('Work dir: ', self.workdir)
            self.images = glob.glob(os.path.join(self.workdir, classname, '*.jpg'))
            print('# Images: ', len(self.images))

        print(self.images[self.idx])
        self.parent.title('Label Helper - {}'.format(self.images[self.idx]))

        self.img = ImageTk.PhotoImage(Image.open(self.images[self.idx]))
        self.panel = tkinter.Label(self.parent, image=self.img)
        self.panel.pack(side='bottom', fill='both', expand='yes')

        # self.buttonframe = tkinter.Frame(self.parent)
        # self.buttonframe.grid(row=1, column=len(classes), columnspan=1)
        for item in classes:
            tkinter.Button(self.parent, text=item,
                           command=lambda  name=item: self.on_btn(name)).pack(side=tkinter.LEFT, padx=5,
                                                     pady=5)

        self.parent.bind("<Left>", self.key_left)
        self.parent.bind("<Right>", self.key_right)

    def on_btn(self, class_name):
        print('btn ', class_name)
        cp_from = self.images[self.idx]
        cp_to = os.path.join(self.workdir, class_name,
                           os.path.basename(self.images[self.idx]))
        if cp_from != cp_to:
            print(cp_from, ' -> ', cp_to)
            shutil.move(cp_from, cp_to)
            del self.images[self.idx]
        elif self.idx < len(self.images):
            self.idx += 1
            print(self.idx)
        self.update_image()

        return

    def key_left(self, event):
        if self.idx > 0:
            self.idx -= 1
        print('left ', self.idx)
        self.update_image()

    def key_right(self, event):
        if self.idx < len(self.images):
            self.idx += 1
        print('right ', self.idx)
        self.update_image()

    def update_image(self):
        if self.idx < len(self.images):
            self.parent.title('Label Helper - {}'.format(self.images[self.idx]))
            self.img = ImageTk.PhotoImage(Image.open(self.images[self.idx]))
            self.panel.configure(image = self.img)
            self.panel.image = self.img
            print('upd')

def main():
    window = tkinter.Tk()
    labelHelper = LabelHelper(window)
    labelHelper.init(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    window.geometry('1920x1080')
    window.mainloop()

if __name__ == '__main__':
    main()

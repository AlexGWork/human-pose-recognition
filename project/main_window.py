# -*- coding: utf-8 -*-
"""
@author: Aleksandr
"""
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from PIL import Image, ImageTk
import convnet_processing as conv_proc

class Main(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.init_main()
    
    def init_main(self):
        self.img = 'temp.jpg'
        toolbar = tk.Frame(bd=2)
        toolbar.pack(side = tk.TOP, fill=tk.X)
        btn_file_dialog = tk.Button(toolbar,
                                     text='выбрать файл',
                                     command = self.open_file_dialog,
                                     compound = tk.TOP)
        btn_file_dialog.pack(side = tk.LEFT)
        btn_process = tk.Button(toolbar,
                                     text='обработать',
                                     command = self.process_image,
                                     compound = tk.TOP)
        btn_process.pack(side = tk.LEFT)
        self.img_container = tk.Label()
        self.img_container.pack(fill=tk.BOTH)        
        self.predicted_label = tk.Label(font=("Consolas", 22))
        self.predicted_label.pack(side=tk.BOTTOM, fill=tk.X)
        
    def open_file_dialog(self):
        file_name = fd.askopenfilename(filetypes=[('JPG Image','*jpg')])
        if not file_name:
            mb.askquestion(title = 'Предупреждение',
                           message = 'Новое изображение не выбрано')
            return
        self.resize_img(file_name)
        
    def resize_img(self, path: str, width = 600, height=480):
        print('sss')
        original_image = Image.open(path)
        original_image = original_image.convert("RGB")
        original_image = original_image.resize((width, height), Image.LANCZOS)
        original_image.save(self.img)
        self.load_img(self.img)        
        
    def load_img(self, path):
        img = ImageTk.PhotoImage(Image.open(self.img))
        self.img_container.configure(image=img)
        self.img_container.image = img
        
    def process_image(self):
        self.predicted_label.configure(text = conv_proc.process(self.img))
        
        
if __name__ == "__main__":
    root = tk.Tk()
    app = Main(root)
    app.pack()
    root.title("MPII classification")
    root.geometry("650x550+400+200")
    root.resizable(False, False)
    root.mainloop()

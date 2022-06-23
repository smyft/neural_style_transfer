import tkinter as tk
from PIL import Image, ImageTk
from NST import neural_style_transfer
import os

window = tk.Tk()
window.title("Image Style Transformer")
window.geometry("900x800")

data_dir = os.path.join(os.path.dirname(__file__), 'data')
content_img_dir = os.path.join(data_dir, 'content_images')
style_img_dir = os.path.join(data_dir, 'style_images')
output_img_dir = os.path.join(data_dir, 'output_images')


def resize_img(target_w, target_h, img):
    w, h = img.size
    scale1 = 1.0 * target_w / w
    scale2 = 1.0 * target_h / h
    scale = min(scale1, scale2)
    result_w = int(w * scale)
    result_h = int(h * scale)
    return img.resize((result_w, result_h), Image.ANTIALIAS)


def show_content_img():
    global content_photo
    global content_path, content_name
    content_name = content_name_entry.get()
    content_path = os.path.join(content_img_dir, content_name)
    img = Image.open(content_path)
    img = resize_img(300, 200, img)
    content_photo = ImageTk.PhotoImage(img)
    content_img_show_label = tk.Label(content_img_frame, image=content_photo)
    content_img_show_label.place(x=0, y=0)
    window.mainloop()


def style():
    if is_style.get() == 1:
        def show_style_img():
            global style_photo
            global style_path, style_name
            style_name = style_name_entry.get()
            style_path = os.path.join(style_img_dir, style_name)
            img = Image.open(style_path)
            img = resize_img(300, 200, img)
            style_photo = ImageTk.PhotoImage(img)
            style_img_show_label = tk.Label(style_img_frame, image=style_photo)
            style_img_show_label.place(x=0, y=0)
            window.mainloop()

        style_name_label = tk.Label(window, font=('Bahnschrift Light', 12), text="风格图名称")
        style_name_label.place(x=25, y=225)

        style_name_entry = tk.Entry(window, width=40, font=('Bahnschrift Light', 12))
        style_name_entry.place(x=25, y=250)

        style_img_label = tk.Label(window, font=('Bahnschrift Light', 12), text="您输入的风格图")
        style_img_label.place(x=575, y=325)
        style_img_button = tk.Button(window, font=('Bahnschrift Light', 12), text='预览', command=show_style_img)
        style_img_button.place(x=700, y=325)
        style_img_frame = tk.Frame(window, bd=1, bg='white', relief='sunken', height=200, width=300)
        style_img_frame.place(x=575, y=375)
        window.mainloop()


def show_output_img():
    output_path = ""
    if is_style.get() == 0:
        default_style_img_name = "default_style_image.jpg"
        output_path = neural_style_transfer(content_name, default_style_img_name, is_preserve_color=is_preserve_color.get())
    else:
        output_path = neural_style_transfer(content_name, style_name, is_preserve_color=is_preserve_color.get())
    global output_photo
    img = Image.open(output_path)
    img = resize_img(400, 400, img)
    output_photo = ImageTk.PhotoImage(img)
    output_img_show_label = tk.Label(output_img_frame, image=output_photo)
    output_img_show_label.place(x=0, y=0)
    window.mainloop()


if __name__ == "__main__":
    title = tk.Label(window, font=('Bahnschrift', 40), text="图像风格迁移（水墨）")
    title.place(x=25, y=25)
    authors = tk.Label(window, font=('Bahnschrift Light', 15), text="宋铭宇 罗逸龙")
    authors.place(x=125, y=90)

    content_name_label = tk.Label(window, font=('Bahnschrift Light', 12), text="内容图名称")
    content_name_label.place(x=25, y=150)

    content_name_entry = tk.Entry(window, width=40, font=('Bahnschrift Light', 12))
    content_name_entry.place(x=25, y=175)

    content_img_label = tk.Label(window, font=('Bahnschrift Light', 12), text='您输入的内容图')
    content_img_label.place(x=575, y=25)
    content_img_button = tk.Button(window, font=('Bahnschrift Light', 12), text='预览', command=show_content_img)
    content_img_button.place(x=700, y=25)
    content_img_frame = tk.Frame(window, bd=1, bg='white', relief='sunken', height=200, width=300)
    content_img_frame.place(x=575, y=75)

    is_style = tk.IntVar()
    style_checkbutton = tk.Checkbutton(window, font=('Bahnschrift Light', 12), text='是否指定一张水墨画作为风格图', variable=is_style,
                                       onvalue=1, offvalue=0, command=style)
    style_checkbutton.place(x=25, y=200)

    is_preserve_color = tk.IntVar()
    preserve_color_checkbutton = tk.Checkbutton(window, font=('Bahnschrift Light', 12), variable=is_preserve_color,
                                                text='是否保留内容图颜色', onvalue=1, offvalue=0)
    preserve_color_checkbutton.place(x=300, y=300)
    output_img_label = tk.Label(window, font=('Bahnschrift Light', 12), text='风格转换结果')
    output_img_label.place(x=25, y=300)
    output_img_button = tk.Button(window, font=('Bahnschrift Light', 12), text='风格转换', command=show_output_img)
    output_img_button.place(x=150, y=300)
    output_img_frame = tk.Frame(window, bd=1, bg='white', relief='sunken', height=300, width=400)
    output_img_frame.place(x=25, y=350)

    window.mainloop()

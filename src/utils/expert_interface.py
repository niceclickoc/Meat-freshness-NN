import tkinter as tk
from PIL import Image, ImageTk

def expert_interface(image_path, file_name, model_prediction, update_prediction_callback):
    root = tk.Tk()
    root.title("Помощь эксперта")
    root.geometry("500x500")

    img = Image.open(image_path)
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)

    image_label = tk.Label(root, image=img_tk)
    image_label.pack(pady=10)

    file_label = tk.Label(root, text=f"Файл: {file_name}")
    file_label.pack(pady=5)

    prediction_label = tk.Label(root, text=f"Модели считают: {model_prediction}")
    prediction_label.pack(pady=5)

    button_label = tk.Label(root, text="Выберите так ли это, или опровергните предсказание")
    button_label.pack(pady=5)


    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    def set_fresh():
        update_prediction_callback(0)
        root.destroy()

    def set_half_fresh():
        update_prediction_callback(1)
        root.destroy()

    def set_spoiled():
        update_prediction_callback(2)
        root.destroy()

    button_fresh = tk.Button(button_frame, text="Fresh", command=set_fresh)
    button_fresh.grid(row=0, column=0, padx=5)

    button_half_fresh = tk.Button(button_frame, text="Half-Fresh", command=set_half_fresh)
    button_half_fresh.grid(row=0, column=1, padx=5)

    button_spoiled = tk.Button(button_frame, text="Spoiled", command=set_spoiled)
    button_spoiled.grid(row=0, column=2, padx=5)

    root.mainloop()
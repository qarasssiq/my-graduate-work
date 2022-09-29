import matplotlib.pyplot as plt
import numpy as np
import math as math
import scipy.special as special
from tkinter import *

funcPlot = plt.figure()


def get_input_field(r, alpha, q, r0, k):
    P = np.where(r - r0 < 0, 0., np.exp(-1j * (k * alpha * (r - r0)) ** q))

    return P


def plot_input_field():
    R = float(txt_r_max1.get())  # макс. значение, которое может принять r
    N = int(txt_N1.get())  # кол-во точек на одной оси, размерность матрицы
    h = 2 * R / (N - 1)  # длина шага по оси x или y
    r = np.linspace(0, 1, 1000)  # r ∈ [0 мм, 1 мм]

    wave_length = float(txt_lambda.get())
    alpha = float(txt_alpha.get())
    q = float(txt_q.get())
    r0 = float(txt_r0.get())
    k = 2 * np.pi / wave_length

    P = get_input_field(r, alpha, q, r0, k)  # функция входного поля
    abs_min = abs(min(np.angle(P)))
    amp = np.abs(P)  # амплитуда
    phase = np.angle(P)  # фаза

    plt.plot(r, amp)
    plt.ylabel('Амплитуда', fontsize=16)
    plt.xlabel('r, мм', fontsize=16)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.grid()
    plt.show()

    plt.plot(r, phase)
    plt.ylabel('Фаза', fontsize=16)
    plt.xlabel('r, мм', fontsize=16)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.grid()
    plt.show()

    x_arr = y_arr = np.linspace(-R, R, N)
    ax_x, ax_y = np.meshgrid(x_arr, y_arr)

    P_matrix = np.angle(get_input_field(np.sqrt(ax_x ** 2 + ax_y ** 2), alpha, q, r0, k)) + abs_min

    plt.imshow(P_matrix, extent=[-R, -R + N * h, -R, -R + N * h],
               cmap='hot', interpolation='nearest')
    plt.xlabel('x, мм', fontsize=16)
    plt.ylabel('y, мм', fontsize=16)

    plt.show()


def get_distribution_function(z, alpha, q, r0, a, b, n, k):
    rectangle_sum = 0.0
    width = (b - a) / n

    for r in np.linspace(a, b, n):
        rectangle_sum += np.where(r - r0 < 0, 0.,
                                  np.exp((-1j * (k * alpha * (r - r0)) ** q + ((1j * k * r ** 2) / (2 * z)))) * r)

    F = (1j * k / z) * np.exp(1j * k * z) * width * rectangle_sum

    return F


def plot_distribution_function_intensity():
    R = float(txt_r_max2.get())  # макс. значение, которое может принять r
    alpha = float(txt_alpha.get())
    z = np.linspace(int(txt_z1.get()), int(txt_z2.get()), 1000)

    wave_length = float(txt_lambda.get())
    q = float(txt_q.get())
    r0 = float(txt_r0.get())
    a = 0.0  # нижний предел интегрирования
    b = R  # верхний предел интегрирования
    n = 1000  # кол-во участков
    k = 2 * np.pi / wave_length

    distribution_function = get_distribution_function(z, alpha, q, r0, a, b, n, k)
    intensity = abs(distribution_function) ** 2

    plt.plot(z, intensity)
    plt.ylabel('Интенсивность, Вт/м2', fontsize=16)
    plt.xlabel('z, мм', fontsize=16)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.grid(True)
    plt.show()


def get_output_field_for_graphs(p, alpha, q, r0, a, b, n, k, z0):
    rectangle_sum = 0.0
    width = (b - a) / n

    for r in np.linspace(a, b, n):
        rectangle_sum += np.where(r - r0 < 0, 0., np.exp(-1j * (k * alpha * (r - r0)) ** q) * np.exp(
            (1j * k * r ** 2) / (2 * z0)) * special.jv(0, (k * r * p) / z0) * r)

    return ((1j * k) / z0) * np.exp(1j * k * z0) * np.exp((1j * k * p ** 2) / (2 * z0)) * width * rectangle_sum


def get_output_field_for_image(p, alpha, q, r0, a, b, n, k, z0):
    rectangle_sum = 0.0
    width = (b - a) / n

    for r in np.linspace(a, b, n):
        rectangle_sum += np.where(r - r0 < 0, 0., np.exp(-1j * (k * alpha * (r - r0)) ** q) * np.exp(
            (1j * k * r ** 2) / (2 * z0)) * special.jv(0, (k * r * p) / z0) * r)

    return (1j * k / z0) * np.exp(1j * k * z0) * np.exp((1j * k * p ** 2) / (2 * z0)) * width * rectangle_sum


def plot_output_field():
    P = float(txt_p_max.get())  # макс. значение, которое может принять p
    p = np.linspace(0, P, 1000)
    N = int(txt_N2.get())  # кол-во точек на одной оси/размерность матрицы/разрешение изображения
    axis_max = P / math.sqrt(2)  # макс. значение, которое могут принять x и y
    h = 2 * axis_max / (N - 1)  # шаг по оси x или y

    wave_length = float(txt_lambda.get())
    alpha = float(txt_alpha.get())
    q = float(txt_q.get())
    r0 = float(txt_r0.get())
    a = 0.0  # нижний предел интегрирования
    b = 1.0  # верхний предел интегрирования
    n = 1000  # кол-во участков
    k = 2 * np.pi / wave_length
    z0 = float(txt_z0.get())

    F = get_output_field_for_graphs(p, alpha, q, r0, a, b, n, k, z0)
    phase = np.angle(F)
    intensity = abs(F) ** 2

    plt.plot(p, intensity)
    plt.ylabel('Интенсивность, Вт/м2', fontsize=16)
    plt.xlabel('p, мм', fontsize=16)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.grid(True)
    plt.show()

    plt.plot(p, phase)
    plt.ylabel('Фаза', fontsize=16)
    plt.xlabel('p, мм', fontsize=16)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.grid(True)
    plt.show()

    x_arr = y_arr = np.linspace(-axis_max, axis_max, N)
    ax_x, ax_y = np.meshgrid(x_arr, y_arr)

    F_matrix = abs(get_output_field_for_image(np.sqrt(ax_x ** 2 + ax_y ** 2), alpha, q, r0, a, b, n, k, z0)) ** 2

    plt.imshow(F_matrix, extent=[-axis_max, -axis_max + 127 * h, -axis_max, -axis_max + 127 * h], cmap='hot',
               interpolation='nearest')

    plt.xlabel('x, мм', fontsize=16)
    plt.ylabel('y, мм', fontsize=16)

    plt.show()


def plot_graphs():
    if selected.get() == 1:
        plot_input_field()
    elif selected.get() == 2:
        plot_distribution_function_intensity()
    elif selected.get() == 3:
        plot_output_field()


def updateRadioButtonVariable(radioButtonVar, updateString):
    radioButtonVar.set(updateString)


window = Tk()

window.title('ВКР')

window.geometry("630x275")

lbl_parameters = Label(window, text="Общие\nпараметры", font=("Arial Bold", 10))
lbl_parameters.grid(column=0, columnspan=2, row=0, sticky="E", padx=10)

selected = IntVar()
selected.set(1)
rad1 = Radiobutton(window, text='Входная функция', variable=selected, value=1, font=("Arial Bold", 10),
                   command=lambda: updateRadioButtonVariable(selected, 1))
rad2 = Radiobutton(window, text='Функция распределения', variable=selected, value=2, font=("Arial Bold", 10),
                   command=lambda: updateRadioButtonVariable(selected, 2))
rad3 = Radiobutton(window, text='Выходная функция', variable=selected, value=3, font=("Arial Bold", 10),
                   command=lambda: updateRadioButtonVariable(selected, 3))
rad1.grid(column=2, columnspan=2, row=0, padx=10, pady=10, sticky="E")
rad2.grid(column=4, columnspan=3, row=0)
rad3.grid(column=7, columnspan=2, row=0)

lbl_lambda = Label(window, text="λ, мм", font=("Arial Bold", 10))
lbl_lambda.grid(column=0, row=1, padx=10)
txt_lambda = Entry(window, width=10)
txt_lambda.grid(column=1, row=1)
txt_lambda.insert(END, '0.00065')

lbl_q = Label(window, text="q", font=("Arial Bold", 10))
lbl_q.grid(column=0, row=2, pady=10, padx=10)
txt_q = Entry(window, width=10)
txt_q.grid(column=1, row=2)
txt_q.insert(END, '1')

lbl_alpha = Label(window, text="α", font=("Arial Bold", 10))
lbl_alpha.grid(column=0, row=3, pady=10, padx=10)
txt_alpha = Entry(window, width=10)
txt_alpha.grid(column=1, row=3)
txt_alpha.insert(END, '0.005')

lbl_r0 = Label(window, text="r0, мм", font=("Arial Bold", 10))
lbl_r0.grid(column=0, row=4, pady=10, padx=10)
txt_r0 = Entry(window, width=10)
txt_r0.grid(column=1, row=4)
txt_r0.insert(END, '0.25')

lbl_r_max1 = Label(window, text="r_max, мм", font=("Arial Bold", 10))
lbl_r_max1.grid(column=2, row=1, pady=10, padx=10)
txt_r_max1 = Entry(window, width=10)
txt_r_max1.grid(column=3, row=1)
txt_r_max1.insert(END, '1')

lbl_N1 = Label(window, text="N", font=("Arial Bold", 10))
lbl_N1.grid(column=2, row=2, pady=10, padx=10)
txt_N1 = Entry(window, width=10)
txt_N1.grid(column=3, row=2)
txt_N1.insert(END, '2056')

lbl_r_max2 = Label(window, text="r_max, мм", font=("Arial Bold", 10))
lbl_r_max2.grid(column=4, row=1, pady=10)
txt_r_max2 = Entry(window, width=10)
txt_r_max2.grid(column=5, columnspan=2, row=1)
txt_r_max2.insert(END, '1')

lbl_z = Label(window, text="z ∈", font=("Arial Bold", 10))
lbl_z.grid(column=4, row=2, pady=10)
txt_z1 = Entry(window, width=10)
txt_z1.grid(column=5, row=2)
txt_z1.insert(END, '3')
txt_z2 = Entry(window, width=10)
txt_z2.grid(column=6, row=2)
txt_z2.insert(END, '1000')

lbl_p_max = Label(window, text="p_max, мм", font=("Arial Bold", 10))
lbl_p_max.grid(column=7, row=1, pady=10)
txt_p_max = Entry(window, width=10)
txt_p_max.grid(column=8, row=1)
txt_p_max.insert(END, '1')

lbl_N2 = Label(window, text="N", font=("Arial Bold", 10))
lbl_N2.grid(column=7, row=2, pady=10)
txt_N2 = Entry(window, width=10)
txt_N2.grid(column=8, row=2)
txt_N2.insert(END, '128')

lbl_z0 = Label(window, text="z0, мм", font=("Arial Bold", 10))
lbl_z0.grid(column=7, row=3, pady=10)
txt_z0 = Entry(window, width=10)
txt_z0.grid(column=8, row=3)
txt_z0.insert(END, '110')

plot_button = Button(master=window,
                     command=plot_graphs,
                     height=2,
                     width=15,
                     text="Построить")

plot_button.grid(column=0, columnspan=9, row=5)

window.mainloop()

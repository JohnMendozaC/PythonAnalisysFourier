
"""

* Universidad Distrital Fransico José de Caldas

* Ingeniería en telemática

* Curso: - Teoría de la información.
         - Codificar señan de digital a digital
         - Codificar análisis de Fourier

* Authors: 

@ - Carol Lizeth Mendoza
@ - John Brayan Mendoza 
@ - Laura Ximena Ahumada
@ - Maria del Carmen Suarez

* Bogotá

* 2021

"""

from types import prepare_class
import sys
import logging
from tkinter import *
from tkinter import ttk
from ftplib import FTP
import os
from datetime import *
from tkinter import filedialog
from tkinter import messagebox
from ipywidgets.widgets.widget_int import Play
import pyaudio
import wave
from playsound import playsound
import matplotlib.pyplot as plt # matplot lib is the premiere plotting lib for Python: https://matplotlib.org/
import numpy as np # numpy is the premiere signal handling library for Python: http://www.numpy.org/
import scipy as sp # for signal processing
from scipy import signal
from scipy.io import wavfile
from scipy.spatial import distance
import IPython.display as ipd
import ipywidgets
import random
import math
import makelab
from makelab import audio
from makelab import signal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from winsound import *
import librosa
import sklearn
import sklearn.tree
import sklearn.tree._utils

def main():
    root = Tk()
    root.configure(background="Silver")
    appClass = MainWindowClass(root)
    root.mainloop()

class MainWindowClass():
    def __init__(self,  master) -> None:
        self.master = master
        self.master.geometry('500x300+100+100')
        self.master.title(" Trabajo final Teoría de la información]")
        self.InitFrame = Frame(self.master)
        self.InitFrame.pack()

  
        """
        *************
        MENU OPCIONES
        *************
        """
        menubar = Menu(self.InitFrame)

        ArchivoMenu_1 = Menu(menubar, tearoff=0)
        ArchivoMenu_1.add_command(label="Salir", command=self.salir)
        menubar.add_cascade(label="Archivo", menu=ArchivoMenu_1)
        self.master.config(menu=menubar) 
        menubar.add_separator()
   
        AudiosMenu_1 = Menu(menubar, tearoff=0)
        AudiosMenu_1.add_command(label="Análisis de Fourier", command=lambda: self.newWindow(AnalisisFourier))
        AudiosMenu_1.add_command(label="Digital a digital", command=lambda: self.newWindow(SenalDigital))
        menubar.add_cascade(label="Audios", menu=AudiosMenu_1)
        self.master.config(menu=menubar)

    """
    ****************************************
    FUNCIONES PARA MANEJO DE NUEVAS VENTANAS
    ****************************************
    """

    def salir(self):
        sys.exit()

    def newWindow(self, _class):
        #global 
        
        try:
            if _class == AnalisisFourier:
                if AnalisisFourier.state()=="normal":
                    AnalisisFourier.focus()
        except:
            newAnalisisFourier = Toplevel(self.master)
            AnalisisFourier(newAnalisisFourier)
        
        
        try:
            if _class == SenalDigital:
                if SenalDigital.state()=="normal":
                    SenalDigital.focus()
        except:
            newSenalDigital = Toplevel(self.master)
            SenalDigital(newSenalDigital)

    def close_window(self):
        self.master.destroy()

"""
**************************************
CLASE PARA MANEJO DE AUDIOS
**************************************
"""

class SenalDigital():
    def __init__(self, master):
        self.master = master
        self.master.geometry('1024x768+300+100')
        self.master.title("Señal digital a digital")

    # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self.master, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
        canvas = Canvas(self.master, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.frmdigital = interior = Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())
        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)

        self.audioname = StringVar(self.frmdigital, value="audio_01.wav")
        self.duracionaudio = StringVar(self.frmdigital, value="0.2")

        self.rootDir = os.getcwd() 

        if os.path.exists(self.rootDir + '\\logs\\proceso.log'):
            pass #os.remove(self.rootDir + '\\logs\\proceso.log')

        logging.basicConfig(filename=self.rootDir + '\\logs\\proceso.log', level=logging.INFO)
        logging.info("INICIO Grabar audio: " + str(datetime.today()))

        lblCustodios = Label(self.frmdigital, text="Bienvenid@ al módulo de grabación.", anchor="e", justify=LEFT)
        lblCustodios.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        lblNombreArchivo = Label(self.frmdigital, text="Nombre Archivo: ", anchor="e", justify=LEFT)
        lblNombreArchivo.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        txtArchivo = Entry(self.frmdigital, textvariable=self.audioname)
        txtArchivo.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        lblDuracion = Label(self.frmdigital, text="Duración audio: ", anchor="e", justify=LEFT)
        lblDuracion.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        txtDuracion = Entry(self.frmdigital, textvariable=self.duracionaudio)
        txtDuracion.grid(row=2, column=1, padx=10, pady=10, sticky="w")
  
        btnInciar= Button(self.frmdigital, text="Iniciar Grabación", command=self.grabar)
        btnInciar.grid(row=3, column=0, padx=10, pady=10, sticky="w")

        self.progress = ttk.Progressbar(self.frmdigital, orient=HORIZONTAL, mode='determinate', length=280 )
        self.progress.grid(row=4,column=0, padx=10, pady=10, columnspan=3, sticky="w")

        self.lbllog = Label(self.frmdigital, text="")
        self.lbllog.grid(row=5, column=0, padx=10, pady=10, sticky="w")

    def btnBuscarExcecute(self):
        filedialog.askopenfilename()
        filedialog.asksaveasfile()

    def grabar(self):
        #DEFINIMOS PARAMETROS
        FORMAT=pyaudio.paInt16
        CHANNELS=2
        RATE=44100
        CHUNK=1024
        duracion=float(self.duracionaudio.get())
        archivo=self.audioname.get()
        
        #INICIAMOS "pyaudio"
        audio=pyaudio.PyAudio()

        #INICIAMOS GRABACIÓN
        stream=audio.open(format=FORMAT,channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

        self.lbllog["text"] = "Grabando audio"
        self.lbllog.update()
        frames=[]
        max = int(RATE/CHUNK*duracion)
        self.progress.configure(maximum=max)

        for i in range(0, max):
            data=stream.read(CHUNK)
            frames.append(data)
            self.progress["value"] = i
            self.progress.update()

            if i%10 == 0:
                self.lbllog["text"] = "Grabando audio .-"
            
            if i%15 == 0:
                self.lbllog["text"] = "Grabando audio ./"

            self.lbllog.update()

        self.lbllog["text"] = "Grabación: terminada. "
        self.progress["value"] = 0
        self.progress.update()

        #DETENEMOS GRABACIÓN
        stream.stop_stream()
        stream.close()
        audio.terminate()

        #CREAMOS/GUARDAMOS EL ARCHIVO DE AUDIO
        waveFile = wave.open(archivo, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        self.digitaltodigital(archivo)


    def digitaltodigital(self, audio=None):
        Bytes = np.fromfile(audio, dtype = "uint8")
        Bits = np.unpackbits(Bytes)
        print(len(Bits))
        
        L = 32 # number of digital samples per data bit
        voltageLevel = 5 # peak voltage level in Volts
        data = Bits
        clk = np.arange(0,2*len(data)) % 2 # clock samples
        
        countZero = 0 # Count zeros in the signal
        countOne = 0 # Count ones in the signal
        previousOne = -5 # Previous one representation
        hdb3 = data * 5 # the inicial HBD3 is just a copy of the digital data multiplied by 5
        
        self.lbllog["text"] = "Procesando señal HDB3 0 %"

        self.lbllog.update()
        val_ = 0

        _max = len(hdb3)

        self.progress.configure(maximum=_max)

        for i in range(0,_max): # from position 0 to last position on hdb3
            if (countZero < 4): # Valid if there are no 4 consecutive zeros
                if (hdb3[i] == 5): # validates if the input position is 5
                    hdb3[i] = previousOne * -1 # if the voltage is different from zero, assigns the opposite value of previousOne to that position 
                    previousOne = hdb3[i] # assigns to previuosOne the new value
                    countOne += 1 # increases the counter of ones
                    countZero = 0 # restarts the zero counter
                if (hdb3[i] == 0): # validates if the input position is 0
                    countZero += 1 # counts consecutive zeros
            if (countZero == 4) and (countOne % 2 == 0): # validating if conutOne is even and there are 4 consecutive zeroes
                hdb3[i-1] = previousOne # the violation is assigned to the previous position
                countZero = 0 # restarts the zero counter
                countOne += 1 # increases the counter of ones
            elif (countZero == 4) and (countOne % 2 != 0): # validating if odd
                previousOne = previousOne * -1 # inverts the value of previousOne to represent the filler
                hdb3[i-3] = previousOne # assigns a filler
                hdb3[i] = hdb3[i-3] # assigns a violation 
                countZero = 0 # restarts the zero counter
            val_ = val_ + 1
            self.progress["value"] = val_
            _percent = (_max / 100) * val_
            _percent = _percent%1024000
            self.progress.update()
            self.lbllog["text"] = "Procesando señal HDB3 {} de {} ".format(str(val_), str(_max))
            self.lbllog.update()

        self.lbllog["text"] = "Procesando señal HDB3 100 % "
        self.lbllog.update()
            
        clk_seq = np.repeat(clk,L) 
        data_seq = np.repeat(data,2*L)
        hdb3_seq = np.repeat(hdb3,2*L)
            
        fig, ax = plt.subplots(3,1,sharex='col', figsize=(10, 8))
        ax[0].plot(clk_seq[0:1000]);ax[0].set_title('Clocking')
        ax[1].plot(data_seq[0:1000]);ax[1].set_title('Digital Data')
        ax[2].plot(hdb3_seq[0:1000]); ax[2].set_title('HDB3')
        
        canvas = FigureCanvasTkAgg(fig, master=self.frmdigital)  # CREAR AREA DE DIBUJO DE TKINTER.
        canvas.draw()
        #canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        canvas.get_tk_widget().grid(row=8, column=0)
        self.frmdigital.update()
        
        #self.stdout_text = tk.Text(
            #self,  bg="black",  fg="#38B179",  font=("Helvetica", 15))
        #self.stdout_text.pack(expand=True, fill=tk.BOTH)
        #sys.stdout = StdOutRedirect(self.stdout_text)

class AnalisisFourier():
    def __init__(self, master) -> None:

        self.master = master
        self.master.geometry('1024x768+300+100')
        self.master.title("Análisis de Fourier")

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self.master, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
        canvas = Canvas(self.master, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.frmanalisis = interior = Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())
        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)

        #self.frmanalisis = Frame(self.master)
        #self.frmanalisis.pack()

        self.audioname = StringVar(self.frmanalisis, value="audio_01.wav")

        self.rootDir = os.getcwd() 

        if os.path.exists(self.rootDir + '\\logs\\proceso.log'):
            pass #os.remove(self.rootDir + '\\logs\\proceso.log')

        logging.basicConfig(filename=self.rootDir + '\\logs\\proceso.log', level=logging.INFO)
        logging.info("INICIO analisis de Fourier: " + str(datetime.today()))

        lblCustodios = Label(self.frmanalisis, text="Bienvenid@ al módulo de análisis de Fourier.", anchor="e", justify=LEFT)
        lblCustodios.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        lblNombreArchivo = Label(self.frmanalisis, text="Nombre Archivo: ", anchor="e", justify=LEFT)
        lblNombreArchivo.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        txtArchivo = Entry(self.frmanalisis, textvariable=self.audioname)
        txtArchivo.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        btnInciar= Button(self.frmanalisis, text="Abrir", command=self.btnBuscarExcecute)
        btnInciar.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.lbllog = Label(self.frmanalisis, text="")
        self.lbllog.grid(row=3, column=0, padx=10, pady=10, sticky="w")

        self.txtLog = Text(self.frmanalisis, height=17, width=70)
        self.txtLog.grid(row=4, column=0, padx=10, pady=10, sticky="w", columnspan=5)

        self.lblprogreso = Label(self.frmanalisis, text="")
        self.lblprogreso.grid(row=5, column=0, padx=10, pady=10, sticky="w")

        self.progress = ttk.Progressbar(self.frmanalisis, orient=HORIZONTAL, mode='determinate', length=280 )
        self.progress.grid(row=5,column=1, padx=10, pady=10, columnspan=3, sticky="w")

    def btnBuscarExcecute(self):
        
        #self.filename = filedialog.askopenfilename()
        self.filename = self.audioname.get()
        self.CodificarAudio(self.filename)

    def CodificarAudio(self, audio=None):
        #Aqui por favor codificar el plot
        audio_data, sampling_rate = librosa.load(self.filename, sr=8000)
        self.txtLog.insert(END, f"Sampling rate: {sampling_rate} Hz\n")
        self.frmanalisis.update()
        self.txtLog.insert(END, f"Number of channels = {len(audio_data.shape)}\n")
        self.frmanalisis.update()
        self.txtLog.insert(END, f"Total samples: {audio_data.shape[0]}")
        self.frmanalisis.update()
        self.txtLog.insert(END, f"Total time: {audio_data.shape[0] / sampling_rate} secs\n")
        self.frmanalisis.update()

        #wm_title(f"{self.filename} at  {sampling_rate} Hz")

        #------------------------------CREAR GRAFICA---------------------------------

        fig, axes = plt.subplots(1, 1, figsize=(4,4))
        #makelab.signal.plot_signal_to_axes(axes, audio_data, sampling_rate)

        ab = self.calc_and_plot_xcorr_dft_with_ground_truth(audio_data, sampling_rate,time_domain_graph_title ="")

        canvas = FigureCanvasTkAgg(ab[0], master=self.frmanalisis)  # CREAR AREA DE DIBUJO DE TKINTER.
        canvas.draw()
        #canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        canvas.get_tk_widget().grid(row=6, column=0, columnspan=50)

        #------------------------------REPRODUCIR AUDIO------------------------------

        play = lambda: PlaySound(self.filename, SND_FILENAME)
        button = Button(self.master, text = 'Play', command = play)
        button.pack()
        play()
        #-----------------------------BOTON CERRAR PENDIENTE-------------------------
        #button = Button(self.master, text="Cerrar", command=cerrar)
        #button.pack()
        
    #def cerrar(self):
        #self.master.quit()     
        #self.master.destroy()
        
    def compute_fft(self,s, sampling_rate, n = None, scale_amplitudes = True):
        if n == None: 
            n = len(s)

        fft_result = np.fft.fft(s, n)
        num_freq_bins = len(fft_result)
        fft_freqs = np.fft.fftfreq(num_freq_bins, d = 1 / sampling_rate)
        half_freq_bins = num_freq_bins // 2
        fft_freqs = fft_freqs[:half_freq_bins]
        fft_result = fft_result[:half_freq_bins]
        fft_amplitudes = np.abs(fft_result)

        if scale_amplitudes is True:
            fft_amplitudes = 2 * fft_amplitudes / (len(s))
            
        return (fft_freqs, fft_amplitudes)

    def get_freq_and_amplitude_csv(self,freqs, amplitudes):
        freq_peaks_with_amplitudes_csv = ""
        for i in range(len(freqs)):
            freq_peaks_with_amplitudes_csv += f"{freqs[i]} Hz ({amplitudes[i]:0.2f})"
            if i + 1 is not len(freqs):
                freq_peaks_with_amplitudes_csv += ", "

        return freq_peaks_with_amplitudes_csv

    def get_top_n_frequency_peaks(self,n, freqs, amplitudes, min_amplitude_threshold = None):

        # Use SciPy signal.find_peaks to find the frequency peaks
        # TODO: in future, could add in support for min horizontal distance so we don't find peaks close together
        fft_peaks_indices, fft_peaks_props = sp.signal.find_peaks(amplitudes, height = min_amplitude_threshold)

        freqs_at_peaks = freqs[fft_peaks_indices]
        amplitudes_at_peaks = amplitudes[fft_peaks_indices]

        if n < len(amplitudes_at_peaks):
            ind = np.argpartition(amplitudes_at_peaks, -n)[-n:] # from https://stackoverflow.com/a/23734295
            ind_sorted_by_coef = ind[np.argsort(-amplitudes_at_peaks[ind])] # reverse sort indices
        else:
            ind_sorted_by_coef = np.argsort(-amplitudes_at_peaks)
            return_list = list(zip(freqs_at_peaks[ind_sorted_by_coef], amplitudes_at_peaks[ind_sorted_by_coef]))
        return return_list

    def get_top_n_frequencies(self,n, freqs, amplitudes, min_amplitude_threshold = None):

        #print(amplitudes)
        if min_amplitude_threshold is not None:
            amplitude_indices = np.where(amplitudes >= min_amplitude_threshold)
            amplitudes = amplitudes[amplitude_indices]
            freqs = freqs[amplitude_indices]

        if n < len(amplitudes):
            ind = np.argpartition(amplitudes, -n)[-n:] # from https://stackoverflow.com/a/23734295
            ind_sorted_by_coef = ind[np.argsort(-amplitudes[ind])] # reverse sort indices
        else:
            ind_sorted_by_coef = np.argsort(-amplitudes)
            return_list = list(zip(freqs[ind_sorted_by_coef], amplitudes[ind_sorted_by_coef]))
        return return_list

    def calc_and_plot_xcorr_dft_with_ground_truth(self,s, sampling_rate, time_domain_graph_title = None, xcorr_freq_step_size = None, xcorr_comparison_signal_length_in_secs = 0.5, normalize_xcorr_result = True, include_annotations = True, minimum_freq_amplitude = 0.08, y_axis_amplitude = True, fft_pad_to = None):

        total_time_in_secs = len(s) / sampling_rate

        # Enumerate through frequencies from 1 to the nyquist limit
        # and run a cross-correlation comparing each freq to the signal
        freq_and_correlation_results = [] # tuple of (freq, max correlation result)
        nyquist_limit = sampling_rate // 2

        freq_step_size = 1
        freq_seq_iter = range(1, nyquist_limit)

        if xcorr_freq_step_size is not None:
            freq_seq_iter = np.arange(1, nyquist_limit, xcorr_freq_step_size)
            freq_step_size = xcorr_freq_step_size

        #self.progress.configure(maximum=int(ipywidgets.FloatProgress(value=1, min=1, max=nyquist_limit)))
        #ipd.display(self.progress)
        num_comparisons = 0

        self.progress.configure(maximum=len(freq_seq_iter))
        self.lblprogreso['text']='Analizando frecuencia.'
        self.lblprogreso.update()
        self.progress.update()

        for test_freq in freq_seq_iter:
            self.frmanalisis.update()
            signal_to_test = makelab.signal.create_sine_wave(test_freq, sampling_rate, xcorr_comparison_signal_length_in_secs)
            correlate_result = np.correlate(s, signal_to_test, 'full')
            # Add in the tuple of test_freq, max correlation result value
            freq_and_correlation_results.append((test_freq, np.max(correlate_result)))
            num_comparisons += 1
            self.progress['value'] += freq_step_size

            if test_freq%10==0:
                self.lblprogreso['text']='Analizando frecuencia -'
            elif test_freq%15==0:
                self.lblprogreso['text']='Analizando frecuencia |'

            self.lblprogreso.update()
            self.progress.update()

        # The `freq_and_correlation_results` is a list of tuple results with (freq, correlation result)
        # Unpack this tuple list into two separate lists freqs, and correlation_results
        correlation_freqs, correlation_results = list(zip(*freq_and_correlation_results))

        correlation_freqs = np.array(correlation_freqs)
        correlation_results = np.array(correlation_results)
        correlation_results_original = correlation_results
        cross_correlation_ylabel = "Cross-correlation result"
        if normalize_xcorr_result:
            correlation_results = correlation_results / (nyquist_limit - 1)
            cross_correlation_ylabel += " (normalized)"

        if y_axis_amplitude:
            correlation_results = (correlation_results_original / (nyquist_limit - 1)) * 2
            cross_correlation_ylabel = "Frequency Amplitude"

            # Plotting
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))
            # Plot the signal in the time domain
            makelab.signal.plot_signal_to_axes(axes[0], s, sampling_rate, title = 
            time_domain_graph_title)
            axes[0].set_title(axes[0].get_title(), y=1.2)
            # Plot the signal correlations (our brute force approach)
            axes[1].plot(correlation_freqs, correlation_results)
            axes[1].set_title("Brute force DFT using cross correlation")
            axes[1].set_xlabel("Frequency")
            axes[1].set_ylabel(cross_correlation_ylabel)

        # Plot the "ground truth" via an FFT
        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.magnitude_spectrum.html
        # Use zero padding to obtain a more accurate estimate of the sinusoidal amplitudes
        # See: https://www.mathworks.com/help/signal/ug/amplitude-estimation-and-zeropadding.html
        if fft_pad_to is None:
            fft_pad_to = len(s) * 4
        elif fft_pad_to == 0:
            fft_pad_to = None

        fft_spectrum, fft_freqs, line = axes[2].magnitude_spectrum(s, Fs =  sampling_rate, color='r', pad_to = fft_pad_to)
        if y_axis_amplitude:
            # By default, the magnitude_spectrum plots half amplitudes (perhaps because it's
            # showing only one-half of the full FFT (the positive frequencies). But there is not
            # way to control this to show full amplitudes by passing in a normalization parameter
            # So, instead, we'll do it by hand here (delete the old line and plot the normalized spectrum)
            line.remove()
            fft_spectrum = np.array(fft_spectrum) * 2
            axes[2].plot(fft_freqs, fft_spectrum, color = 'r')
            axes[2].set_ylabel("Frequency Amplitude")
            axes[2].set_title("Using built-in FFT via matplotlib.pyplot.magnitude_spectrum")
        # Find the number of peaks
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html
        # Du et al., Improved Peak Detection, Bioinformatics: https://academic.oup.com/bioinformatics/article/22/17/2059/274284
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        xcorr_dft_peaks_indices, xcorr_dft_peaks_props = sp.signal.find_peaks(correlation_results, height = minimum_freq_amplitude)
        xcorr_freq_peaks = correlation_freqs[xcorr_dft_peaks_indices]
        xcorr_freq_peak_amplitudes = correlation_results[xcorr_dft_peaks_indices]
        xcorr_freq_peaks_csv = ", ".join(map(str, xcorr_freq_peaks))
        xcorr_freq_peaks_with_amplitudes_csv = self.get_freq_and_amplitude_csv(xcorr_freq_peaks, xcorr_freq_peak_amplitudes)
        xcorr_freq_peaks_array_str = [f"{freq} Hz" for freq in xcorr_freq_peaks]

        fft_peaks_indices, fft_peaks_props = sp.signal.find_peaks(fft_spectrum, height = minimum_freq_amplitude)
        fft_freq_peaks = fft_freqs[fft_peaks_indices]
        fft_freq_peaks_amplitudes = fft_spectrum[fft_peaks_indices]
        fft_freq_peaks_csv = ", ".join(map(str, fft_freq_peaks))
        fft_freq_peaks_with_amplitudes_csv = self.get_freq_and_amplitude_csv(fft_freq_peaks, fft_freq_peaks_amplitudes)
        fft_freq_peaks_array_str = [f"{freq} Hz" for freq in fft_freq_peaks]

        # Print out frequency analysis info and annotate plots
        if include_annotations:
            axes[1].plot(xcorr_freq_peaks, xcorr_freq_peak_amplitudes, linestyle='None', marker="x", color="red", alpha=0.8)
            for i in range(len(xcorr_freq_peaks)):
                axes[1].text(xcorr_freq_peaks[i] + 2, xcorr_freq_peak_amplitudes[i], f"{xcorr_freq_peaks[i]} Hz", color="red")
                axes[2].plot(fft_freq_peaks, fft_freq_peaks_amplitudes, linestyle='None', marker="x", color="black", alpha=0.8)
            for i in range(len(fft_freq_peaks)):
                axes[2].text(fft_freq_peaks[i] + 2, fft_freq_peaks_amplitudes[i], f"{fft_freq_peaks[i]} Hz", color="black")

        self.txtLog.insert(END,"**Brute force cross-correlation DFT**\n")
        self.frmanalisis.update()
        self.txtLog.insert(END,f"Num cross correlations: {num_comparisons}\n")
        self.frmanalisis.update()
        self.txtLog.insert(END,f"Frequency step resolution for cross correlation: {freq_step_size}\n")
        self.frmanalisis.update()
        self.txtLog.insert(END,f"Found {len(xcorr_dft_peaks_indices)} freq peak(s) at: {xcorr_freq_peaks_csv} Hz\n")
        self.frmanalisis.update()
        self.txtLog.insert(END,f"The minimum peak amplitude threshold set to: {minimum_freq_amplitude}\n")
        self.frmanalisis.update()
        self.txtLog.insert(END,f"Freq and amplitudes: {xcorr_freq_peaks_with_amplitudes_csv} Hz\n")
        self.frmanalisis.update()
        self.txtLog.insert(END,"\n")
        self.frmanalisis.update()
        self.txtLog.insert(END,"**Ground truth FFT**\n")
        self.frmanalisis.update()
        self.txtLog.insert(END,f"Num FFT freq bins: {len(fft_freqs)}\n")
        self.frmanalisis.update()
        self.txtLog.insert(END,f"FFT Freq bin spacing: {fft_freqs[1] - fft_freqs[0]}\n")
        self.frmanalisis.update()
        self.txtLog.insert(END,f"Found {len(fft_peaks_indices)} freq peak(s) at: {fft_freq_peaks_csv} Hz\n")
        self.frmanalisis.update()
        self.txtLog.insert(END,f"The minimum peak amplitude threshold set to: {minimum_freq_amplitude}\n")
        self.frmanalisis.update()
        self.txtLog.insert(END,f"Freq and amplitudes: {fft_freq_peaks_with_amplitudes_csv} Hz\n")
        self.frmanalisis.update()
        #print(fft_freqs[fft_peaks_indices] + "Hz")

        fig.tight_layout(pad=2)
        return (fig, axes, correlation_freqs, correlation_results, fft_freqs, fft_spectrum)    
   
if __name__ == "__main__":
    try:
        rootDir = os.getcwd()
        rootDir = rootDir + "\\logs"
        if not os.path.exists(rootDir):
            os.mkdir(rootDir)
        logsFile = rootDir + "\\logs.log"

        if os.path.exists(logsFile):
            pass #os.remove(logsFile)

        open(logsFile, "w")
        logging.basicConfig(filename=logsFile, level=logging.INFO)
        main()
    except KeyboardInterrupt:
        sys.exit(1)



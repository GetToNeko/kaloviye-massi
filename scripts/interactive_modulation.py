import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scripts.generate_dataset_improved import (
    generate_modulation, SignalGenerator, add_noise,
    SAMPLE_LENGTH, SAMPLING_RATE
)

# Константы
CARRIER_FREQ_RANGE = (1000, 10000)  # 1kHz to 10kHz
MOD_FREQ_RANGE = (50, 500)  # 50-500 Hz

class ModulationGenerator:
    def __init__(self):
        self.mod_types = {
            '1': ('FM', 'FM'),
            '2': ('AM', 'AM'),
            '3': ('PM', 'PM'),
            '4': ('FSK', 'FSK'),
            '5': ('ASK', 'ASK'),
            '6': ('BPSK', 'BPSK'),
            '7': ('QPSK', 'QPSK')
        }
        
        self.snr_levels = {
            '1': 0,
            '2': 5,
            '3': 10,
            '4': 15,
            '5': 20
        }
        
        self.current_mod_signal = None
        self.current_noisy_signal = None
        self.current_message = None
        self.current_mod_type = None
    
    def print_menu(self):
        print("\nВыберите тип модуляции:")
        for key, (display_name, _) in self.mod_types.items():
            print(f"{key}) {display_name}")
        print("q) Выход")
    
    def print_snr_menu(self):
        print("\nВыберите уровень шума (SNR дБ):")
        for key, snr in self.snr_levels.items():
            print(f"{key}) {snr}")
    
    def generate_signal(self, mod_choice, snr_choice):
        """Генерация сигнала с выбранными параметрами"""
        if mod_choice not in self.mod_types:
            return False
        
        # Получение параметров
        display_name, mod_type = self.mod_types[mod_choice]
        snr = self.snr_levels[snr_choice]
        
        # Генерация временной оси
        t = np.linspace(0, SAMPLE_LENGTH/SAMPLING_RATE, SAMPLE_LENGTH)
        
        # Генерация модулированного сигнала
        complex_signal = generate_modulation(t, mod_type)
        
        # Разделение на I/Q компоненты
        self.current_mod_signal = np.array([np.real(complex_signal), np.imag(complex_signal)])
        
        # Добавление шума
        noisy_complex = add_noise(complex_signal, snr)
        self.current_noisy_signal = np.array([np.real(noisy_complex), np.imag(noisy_complex)])
        
        # Сохраняем модулирующий сигнал (для отображения)
        self.current_message = SignalGenerator.generate_message(t, mode='random_sin')
        
        self.current_mod_type = display_name
        return True
    
    def plot_signals(self):
        """Отображение графиков сигналов"""
        if self.current_mod_signal is None:
            print("Сначала сгенерируйте сигнал!")
            return
        
        fig = plt.figure(figsize=(15, 12))
        
        # График чистого сигнала
        plt.subplot(321)
        plt.plot(self.current_mod_signal[0], label='I')
        plt.plot(self.current_mod_signal[1], label='Q')
        plt.title(f'Чистый сигнал ({self.current_mod_type})')
        plt.legend()
        plt.grid(True)
        
        # График зашумленного сигнала
        plt.subplot(322)
        plt.plot(self.current_noisy_signal[0], label='I')
        plt.plot(self.current_noisy_signal[1], label='Q')
        plt.title('Зашумленный сигнал')
        plt.legend()
        plt.grid(True)
        
        # Констелляционная диаграмма чистого сигнала
        plt.subplot(323)
        plt.scatter(self.current_mod_signal[0], self.current_mod_signal[1], 
                   alpha=0.5, s=1)
        plt.title('Констелляционная диаграмма (чистый)')
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.grid(True)
        
        # Констелляционная диаграмма зашумленного сигнала
        plt.subplot(324)
        plt.scatter(self.current_noisy_signal[0], self.current_noisy_signal[1], 
                   alpha=0.5, s=1)
        plt.title('Констелляционная диаграмма (шум)')
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.grid(True)
        
        # Спектр чистого сигнала
        plt.subplot(325)
        clean_complex = self.current_mod_signal[0] + 1j * self.current_mod_signal[1]
        freq = np.fft.fftfreq(SAMPLE_LENGTH, 1/SAMPLING_RATE)
        spectrum = np.abs(np.fft.fft(clean_complex))
        plt.plot(freq[:SAMPLE_LENGTH//2], spectrum[:SAMPLE_LENGTH//2])
        plt.title('Спектр чистого сигнала')
        plt.xlabel('Частота (Гц)')
        plt.grid(True)
        
        # Спектр зашумленного сигнала
        plt.subplot(326)
        noisy_complex = self.current_noisy_signal[0] + 1j * self.current_noisy_signal[1]
        spectrum = np.abs(np.fft.fft(noisy_complex))
        plt.plot(freq[:SAMPLE_LENGTH//2], spectrum[:SAMPLE_LENGTH//2])
        plt.title('Спектр зашумленного сигнала')
        plt.xlabel('Частота (Гц)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run(self):
        """Основной цикл программы"""
        while True:
            self.print_menu()
            mod_choice = input(">> ").strip().lower()
            
            if mod_choice == 'q':
                break
            
            if mod_choice not in self.mod_types:
                print("Неверный выбор! Попробуйте снова.")
                continue
            
            self.print_snr_menu()
            snr_choice = input(">> ").strip()
            
            if snr_choice not in self.snr_levels:
                print("Неверный выбор! Попробуйте снова.")
                continue
            
            print("\nГенерация сигнала...")
            if self.generate_signal(mod_choice, snr_choice):
                while True:
                    show_plots = input("\nПоказать графики? (y/n) >> ").strip().lower()
                    if show_plots == 'y':
                        self.plot_signals()
                        break
                    elif show_plots == 'n':
                        break
                    else:
                        print("Пожалуйста, введите 'y' или 'n'")

def generate_modulated_signal(mod_type, snr):
    """Генерация модулированного сигнала для заданного типа модуляции и SNR
    
    Args:
        mod_type (str): Тип модуляции (AM, FM, PM, ASK, FSK, BPSK, QPSK)
        snr (int): Отношение сигнал/шум в дБ
        
    Returns:
        tuple: (clean_signal, noisy_signal), где каждый сигнал имеет форму (2, SAMPLE_LENGTH)
    """
    # Генерация временной оси
    t = np.linspace(0, SAMPLE_LENGTH/SAMPLING_RATE, SAMPLE_LENGTH)
    
    # Генерация комплексного сигнала
    complex_signal = generate_modulation(t, mod_type)
    
    # Разделение на I/Q компоненты
    clean_signal = np.array([np.real(complex_signal), np.imag(complex_signal)])
    
    # Добавление шума
    noisy_complex = add_noise(complex_signal, snr)
    noisy_signal = np.array([np.real(noisy_complex), np.imag(noisy_complex)])
    
    return clean_signal, noisy_signal

if __name__ == "__main__":
    generator = ModulationGenerator()
    generator.run()
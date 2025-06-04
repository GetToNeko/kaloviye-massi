import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from generate_dataset import (
    generate_fm, generate_am, generate_pm,
    generate_fsk, generate_ask, generate_bpsk, generate_qpsk,
    generate_time_base, generate_random_bits, add_noise
)

# Константы
SAMPLE_LENGTH = 512
SAMPLING_RATE = 100000  # 100kHz
BIT_LENGTH = 256
CARRIER_FREQ_RANGE = (1000, 10000)  # 1kHz to 10kHz
MOD_FREQ_RANGE = (50, 500)  # 50-500 Hz

class ModulationGenerator:
    def __init__(self):
        self.mod_types = {
            '1': ('FM', generate_fm),
            '2': ('AM', generate_am),
            '3': ('PM', generate_pm),
            '4': ('FSK', generate_fsk),
            '5': ('ASK', generate_ask),
            '6': ('BPSK', generate_bpsk),
            '7': ('QPSK', generate_qpsk)
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
        for key, (mod_type, _) in self.mod_types.items():
            print(f"{key}) {mod_type}")
        print("q) Выход")
    
    def print_snr_menu(self):
        print("\nВыберите уровень шума (SNR дБ):")
        for key, snr in self.snr_levels.items():
            print(f"{key}) {snr}")
    
    def generate_modulating_signal(self, mod_type):
        """Генерация модулирующего сигнала"""
        if mod_type in ['FM', 'AM', 'PM']:
            t = generate_time_base()
            mod_freq = np.random.uniform(*MOD_FREQ_RANGE)
            return np.sin(2 * np.pi * mod_freq * t)
        else:
            return generate_random_bits()
    
    def generate_signal(self, mod_choice, snr_choice):
        """Генерация сигнала с выбранными параметрами"""
        if mod_choice not in self.mod_types:
            return False
        
        # Получение параметров
        mod_type, mod_func = self.mod_types[mod_choice]
        snr = self.snr_levels[snr_choice]
        
        # Генерация несущей частоты
        carrier_freq = np.random.uniform(*CARRIER_FREQ_RANGE)
        
        # Генерация модулирующего сигнала
        self.current_message = self.generate_modulating_signal(mod_type)
        
        # Генерация модулированного сигнала
        i_signal, q_signal = mod_func(carrier_freq)
        self.current_mod_signal = np.array([i_signal, q_signal])
        
        # Добавление шума
        i_noisy, q_noisy = add_noise(i_signal, q_signal, snr)
        self.current_noisy_signal = np.array([i_noisy, q_noisy])
        
        self.current_mod_type = mod_type
        return True
    
    def plot_signals(self):
        """Отображение графиков сигналов"""
        if self.current_mod_signal is None:
            print("Сначала сгенерируйте сигнал!")
            return
        
        fig = plt.figure(figsize=(15, 10))
        
        # График чистого сигнала
        plt.subplot(311)
        plt.plot(self.current_mod_signal[0], label='I')
        plt.plot(self.current_mod_signal[1], label='Q')
        plt.title(f'Чистый сигнал ({self.current_mod_type})')
        plt.legend()
        plt.grid(True)
        
        # График зашумленного сигнала
        plt.subplot(312)
        plt.plot(self.current_noisy_signal[0], label='I')
        plt.plot(self.current_noisy_signal[1], label='Q')
        plt.title('Зашумленный сигнал')
        plt.legend()
        plt.grid(True)
        
        # График модулирующего сигнала
        plt.subplot(313)
        if self.current_mod_type in ['FM', 'AM', 'PM']:
            plt.plot(self.current_message)
            plt.title('Модулирующий сигнал (синусоида)')
        else:
            plt.step(np.arange(len(self.current_message)), self.current_message)
            plt.title('Модулирующий сигнал (битовая последовательность)')
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

if __name__ == "__main__":
    generator = ModulationGenerator()
    generator.run() 

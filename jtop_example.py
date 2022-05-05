from jtop import jtop, JtopException
import numpy as np
import time

repetitions = 100

temps_cpu = np.zeros((repetitions,1))
temps_gpu = np.zeros((repetitions,1))

freq_cpu = np.zeros((repetitions,1))
freq_gpu = np.zeros((repetitions,1))

power_cur = np.zeros((repetitions,1))
power_avg = np.zeros((repetitions,1))

power_cur_gpu = np.zeros((repetitions,1))
power_avg_gpu = np.zeros((repetitions,1))

power_cur_cpu = np.zeros((repetitions,1))
power_avg_cpu = np.zeros((repetitions,1))

with jtop() as jetson:
  for rep in range(repetitions):
    temps_cpu[rep] = jetson.stats['Temp CPU']
    temps_gpu[rep] = jetson.stats['Temp GPU']

    freq_cpu[rep] = (jetson.cpu['CPU1']['frq'] + jetson.cpu['CPU2']['frq'] + jetson.cpu['CPU3']['frq'] + jetson.cpu['CPU4']['frq']) / 4
    freq_gpu[rep] = jetson.gpu['frq']

    power_cur[rep] = jetson.power[0]['cur']
    power_avg[rep] = jetson.power[0]['avg']

    power_cur_gpu[rep] = jetson.power[1]['5V GPU']['cur']
    power_avg_gpu[rep] = jetson.power[1]['5V GPU']['avg']
    power_cur_cpu[rep] = jetson.power[1]['5V CPU']['cur']
    power_avg_cpu[rep] = jetson.power[1]['5V CPU']['avg']

avg_temps_cpu = sum(temps_cpu)[0]/repetitions
avg_temps_gpu = sum(temps_gpu)[0]/repetitions

avg_freq_cpu = sum(freq_cpu)[0]/repetitions
avg_freq_gpu = sum(freq_gpu)[0]/repetitions

avg_power_cur = sum(power_cur)[0]/repetitions
avg_power_avg = sum(power_avg)[0]/repetitions

avg_power_cur_gpu = sum(power_cur_gpu)[0]/repetitions
avg_power_avg_gpu = sum(power_avg_gpu)[0]/repetitions
avg_power_cur_cpu = sum(power_cur_cpu)[0]/repetitions
avg_power_avg_cpu = sum(power_avg_cpu)[0]/repetitions

print(avg_temps_cpu)
print(avg_temps_gpu)

print(avg_freq_cpu)
print(avg_freq_gpu)

print(avg_power_cur)
print(avg_power_avg)

print(avg_power_cur_gpu)
print(avg_power_avg_gpu)
print(avg_power_cur_cpu)
print(avg_power_avg_cpu)
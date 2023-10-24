
from event_generators import gen_arrival_t, gen_observer_t, gen_departure_t, gen_service_t
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import sys

def calc_dep_t(last_departure_time, arrival_time, service_time):
  return max(last_departure_time, arrival_time) + service_time 

def MM1K(avg_arrival_rate, avg_pckt_len, trans_rate, sim_time, K=None):
  rho = avg_pckt_len*avg_arrival_rate/trans_rate #Utilization of the queue (= input rate/service rate = L λ/C)
  En = 0 #Average number of packets in the buffer/queue
  P_idle = 0 #The proportion of time the server is idle
  P_loss = 0 #The packet loss probability (0 for MM1)

  if (K == None):
    print("K cannot be Null")
    sys.exit(1) 

  # generate & sort events by time
  arrivals = gen_arrival_t(avg_arrival=avg_arrival_rate,T=sim_time)
  service_times = gen_service_t(arrivals=arrivals, avg_pckt_len=avg_pckt_len, trans_rate=trans_rate) 
  observers = gen_observer_t(avg_arrival=avg_arrival_rate,T=sim_time)
  
  # Sort events by time
  events = arrivals + observers
  events.sort(key = lambda x:x[1])

  T = events[-1][1]
  N_arrivals = len(arrivals)
  N_observers = len(observers)

  dep_q = []
  q = 0
  last_event_time = 0

  for event in events:
    t = event[1]
      
    while(q > 0 and dep_q[0] <= t):
      dep_q.pop(0)
      q -= 1

    if (event[0] == "observer"):
      dt = t - last_event_time
      # the time-average of the number of packets in the queue
      if ( q == 0):
        P_idle += dt
  
      En += q

    elif (event[0] == "arrival"):
      # add event to Q if space
      # Q not full
      if ( q < K ):
        if q == 0:
          last_dep_t = t
        else:
          last_dep_t = dep_q[-1]
        

        dep_time = calc_dep_t(last_dep_t, t, service_times[0])
        q += 1
        dep_q.append(dep_time)
        
      # Q full
      else:
        P_loss += 1

      service_times.pop(0)

    last_event_time = t
    q = len(dep_q)

  return [rho, En/(N_observers), P_idle/T, P_loss/N_arrivals]

def MM1K_rho(L, C, T, K, rho_start=0.5, rho_end= 2.5, rho_step=0.1):
  
  rho_list, En_list, P_idle_list, P_loss_list = [], [], [], []
  RHO = np.arange(rho_start, rho_end, rho_step)
  print(f"\n############### L = {L} bits/packet, C = {C} bits/sec, T = {T} sec, K = {K} queue size ###############")

  for rho in RHO:
    print(f"\n// rho = {rho} //")
    a = rho*C/L
    rho, En, P_idle, P_loss = MM1K(avg_arrival_rate=a, avg_pckt_len=L, trans_rate=C, sim_time=T, K= K)
    rho_list.append(rho)
    En_list.append(En)
    P_idle_list.append(P_idle)
    P_loss_list.append(P_loss)
    
    print(f"E[n]   =", En)
    print(f"P_idel = ", P_idle)
    print(f"P_loss = ", P_loss)

  return rho_list, En_list, P_idle_list, P_loss_list

def display_MM1K_rho_results(L, C, T, timestamp, K):
  rho_list, En_list, P_idle_list, P_loss_list = [], [], [], []
  N = len(K)

  # Create the directory if it doesn't exist
  directory_name = f"MM1K_of_rho/{timestamp}/"
  file_name = f"{directory_name}/MM1K_rho_K.csv"
  print(f"create: {directory_name}")
  if not os.path.exists(directory_name):
    os.makedirs(directory_name)

  # Write to file
  with open(file_name, "w") as file:
    
    for k in K:
      print(f"\n -- K = {k} -- ")
      # Store K lists data 
      rho_list_k, En_list_k, P_list_idle_k, P_list_loss_k = [],[],[],[]

      # Write CSV header
      line = f"K,Rho,E[n],P_idle,P_loss\n"
      file.write(line)
      
      # Catch results from simulations
      rho_list_k, En_list_k, P_list_idle_k, P_list_loss_k = MM1K_rho(L=L, C=C, T=T, K=k)
      
      
      rho_list.append(rho_list_k)
      En_list.append(En_list_k)
      P_idle_list.append(P_list_idle_k)
      P_loss_list.append(P_list_loss_k)
      
      # Write new datadata to file
      for i in range(len(rho_list_k)):
        line = f"{k},{rho_list_k[i]},{En_list_k[i]},{P_list_idle_k[i]},{P_list_loss_k[i]}\n"
        file.write(line)

  file.close()

  
  en_figname = directory_name+"MM1K_En_rho.png"
  idle_figname = directory_name+"MM1K_Idle_rho.png"
  loss_figname = directory_name+"MM1K_Loss_rho.png"
  
  en_fig, en_ax = plt.subplots(1, 1, figsize=(8, 6))
  idle_fig, idle_ax = plt.subplots(1, 1, figsize=(8, 6))
  loss_fig, loss_ax = plt.subplots(1, 1, figsize=(8, 6))

  
  for i in range(N):
    en_ax.plot(rho_list[i], En_list[i], label=K[i])
    idle_ax.plot(rho_list[i], P_idle_list[i], label=K[i])
    loss_ax.plot(rho_list[i], P_loss_list[i], label=K[i])

    en_ax.set_ylabel('E[n], Avg Queue Size ')
    en_ax.set_xlabel('% Utilization')

    idle_ax.set_ylabel('% Idle')
    idle_ax.set_xlabel('% Utilization')

    loss_ax.set_ylabel('%  Loss')
    loss_ax.set_xlabel('% Utilization')

    en_ax.set_title(f'E[n] v Rho, K = {K[i]}, T= {T} s, L= {L/10**3} kb/pkt, C= {C/10**6} Mbps')
    idle_ax.set_title(f'P_Idle v Rho, K = {K[i]}, T={T}, L={L}, C={C}')
    loss_ax.set_title(f'P_Loss v Rho, K = {K[i]}, T={T}, L={L}, C={C}')

    en_ax.legend()
    idle_ax.legend()
    loss_ax.legend()

  en_fig.savefig(en_figname)
  idle_fig.savefig(idle_figname) 
  loss_fig.savefig(loss_figname)

  plt.show()
  return 0

##########################################


def main():
  """
  input: 
    - a = arrival rate (pkts/sec)
    - s = service rate (pkts/sec)
    - L = avg packet length (bits)
    - C = transmission rate (bits/s)
  """
  # Parameters
  N = 10
  T = 100
  L = 2_000
  C = 1_000_000
  A = 75

  # Timestamp
  timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H.%M.%S_EST")
  print("\nBegin Time: ", timestamp,"\n")
  # Measure Simulation Stability ( Values approach 5% )
  # measure_simulation_stability(N, T, L, C, A, timestamp)

  # Rho in [0.5, 1.5] 0.1 stepsize + K in [10, 25, 50]
  display_MM1K_rho_results(L=L, C=C, T=T, timestamp=timestamp, K=[10, 25, 50])
  # MM1K_rho(L=L, C=C, T=T, K=10)

  return 0 

if __name__ == '__main__':
  main()
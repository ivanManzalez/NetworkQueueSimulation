from event_generators import gen_arrival_t, gen_observer_t, gen_departure_t
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import sys

####################################################################################################

def calc_deltas(N, T, En_list, P_idle_list, P_loss_list):
  d_en = []
  dP_idle = []
  dP_loss = []

  for i in range(1,N-1):
    try:
      d_en.append(100*(En_list[i]-En_list[i-1])/En_list[i-1])
      
    except ZeroDivisionError as e:
      d_en.append(0)

    try:
      dP_idle.append(100*(P_idle_list[i]-P_idle_list[i-1])/P_idle_list[i-1])
      
    except ZeroDivisionError as e:
      dP_idle.append(0)

    try:
      dP_loss.append(100*(P_loss_list[i]-P_loss_list[i-1])/P_loss_list[i-1])
    except ZeroDivisionError as e:
      dP_loss.append(0)
    
  return d_en, dP_idle,dP_loss

def MM1(avg_arrival_rate, avg_pckt_len, trans_rate, sim_time,K=None):
  rho = avg_pckt_len*avg_arrival_rate/trans_rate #Utilization of the queue (= input rate/service rate = L λ/C)
  En = 0 #Average number of packets in the buffer/queue
  P_idle = 0 #The proportion of time the server is idle
  P_loss = 0 #The packet loss probability (0 for MM1)

  if (K == None):
    K = np.inf

  # generate & sort events by time
  arrivals = gen_arrival_t(avg_arrival=avg_arrival_rate,T=sim_time)
  departures = gen_departure_t(avg_pckt_len=avg_pckt_len, trans_rate=trans_rate, arrivals=arrivals)
  observers = gen_observer_t(avg_arrival=avg_arrival_rate*5,  T=sim_time)
  events = arrivals + departures + observers
  events.sort(key=lambda x:x[1])
  T = events[-1][1]
  N_arrivals = len(arrivals)
  N_observers = len(observers)
  # print("\n-- Number of Events: ", len(events), "--")

  event_queue = 0
  last_event_time = 0

  for event in events:
    t = event[1]

    if (event[0] == "observer"):
      dt = t - last_event_time
      # the time-average of the number of packets in the queue
      if (event_queue == 0):
        # print("Time fraction of sys not in use, dt=",dt)
        P_idle += dt
  
      En += event_queue

    elif (event[0] == "arrival"):
      # add event to Q
      if (event_queue < K):
        event_queue += 1
      
      else:
        P_loss += 1
    
    elif(event[0] == "departure"):
      # remove first event from Q
      event_queue -= 1
      # N_departures += 1
    
    last_event_time = t

  return [rho, En/(N_observers), P_idle/T, P_loss/N_arrivals]

def MM1_stability(N, T, L, C,A):
  rho_list, En_list, P_idle_list, P_loss_list = [], [], [], []
  rho = A*L/C
  print(f"\n#################################################################")
  print(f"T={T}s, L={L}bits, C={C}bits/s, A={A}pkts/s")
  print(f"\n#################################################################")
  for n in range(1, N):
    title = f"\n-- T={n*T}s (n={n})--"
    print(title)
    rho, En, P_idle, P_loss = MM1(avg_arrival_rate=A, avg_pckt_len= L, trans_rate=C, sim_time=n*T)
    rho_list.append(rho)
    En_list.append(En)
    P_idle_list.append(P_idle)
    P_loss_list.append(P_loss)
    print("Complete.")

  return calc_deltas(N, T, En_list, P_idle_list, P_loss_list)

def measure_simulation_stability(N, T, L, C, A, timestamp):
  # Data
  dEn, dP_idle, dP_loss = MM1_stability(N, T, L, C, A)

  directory_name = 'MM1_stability'
  file_name = f"{directory_name}/MM1_stability_{timestamp}.png"

  # Create the directory if it doesn't exist
  if not os.path.exists(directory_name):
    os.makedirs(directory_name)

  # Create a figure
  t = np.arange(1, N-1) * T
  fig, ax = plt.subplots(1, 1, figsize=(8, 6))

  # Create a subplot
  ax.plot(t, dEn, color='red', label='dEn')
  ax.plot(t, dP_idle, color='green', label='dP_idle')
  ax.plot(t, dP_loss, color='blue', label='dP_loss')
  

  # Customize the plot
  ax.set_xlabel('T, total simulated time, s')
  ax.set_ylabel('% Deltas')
  ax.set_title(f'T = {T} sec, L = {L} bits, C= {C} bits/sec, A= {A} pkts/sec')
  ax.grid(which='both',axis='both', linestyle=':', linewidth='0.5', color='gray')
  # ax.tick_params(axis='both')
  ax.legend() 

  # save & show 
  plt.savefig(file_name) 
  plt.show()

  return 1

def MM1_rho( L, C, T, K, rho_star=0.25, rho_end=0.85,rho_step=0.1):
  # Function of Rho, Independent Variable
  RHO = np.arange(rho_star, rho_end, rho_step)

  # Data store, Dependent Variables
  rho_list, A_list, En_list, P_idle_list, P_loss_list = [], [], [], [], []

  # Simulate for each rho and 
  # Parameters defined above
  for rho in RHO:
    a = rho*C/L
    print(f"\n#### (rho={rho}, a={a}, T={T}, L={L}, C={C})   ####")
    
    rho, En, P_idle, P_loss = MM1(avg_arrival_rate=a, avg_pckt_len= L, trans_rate=C, sim_time=T, K=K)
    rho_list.append(rho)
    A_list.append(a)
    En_list.append(En)
    P_idle_list.append(P_idle)
    P_loss_list.append(P_loss)

    print("\nTotal elapsed time:",T)
    print("rho    : ", rho)
    print("En     : ",En)
    print("P_idle : ", P_idle)
    print("P_loss : ", P_loss)
  print(f"####################### END #################################")

  return rho_list, A_list, En_list, P_idle_list, P_loss_list

def MM1_rho_results(L, C, T, timestamp, K=[None], rho_star=0.25, rho_end=0.95):
  # Data store
  rho_list, A_list, En_list, P_idle_list, P_loss_list = [], [], [], [], []

  directory_name = 'MM1_of_rho/'+timestamp

  # Create the directory if it doesn't exist
  if not os.path.exists(directory_name):
    os.makedirs(directory_name)

  file_name = f"{directory_name}/MM1_rho_K.csv"
  en_fig = f"{directory_name}/MM1_En_rho_K.png"
  idle_fig = f"{directory_name}/MM1_Idle_rho_K.png"
  loss_fig = f"{directory_name}/MM1_Loss_rho_K.png"
  
  fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
  fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
  fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
  
  for k in range(len(K)):
    
    k_i = K[k]
    print(f"K = {k_i}")
    rho_list, A_list, En_list, P_idle_list, P_loss_list = MM1_rho(L=L, C=C, T=T, K=k_i, rho_star=rho_star, rho_end=rho_end)

    # Write results to file
    with open(file_name, "a") as file:
      file.write("K,Rho,A,En,P_idle,P_loss\n")
      for i in range(len(rho_list)):
        line = f"{K[k]},{rho_list[i]},{A_list[i]},{En_list[i]},{P_idle_list[i]},{P_loss_list[i]}\n"
        file.write(line)
    file.close()
    
    # # Create a subplot
    ax1.plot(rho_list, En_list,label=f'E[n], K={K[k]}')
    ax2.plot(rho_list, P_idle_list, label=f'P_idle, K={K[k]}')
    ax3.plot(rho_list, P_loss_list, label=f'P_loss, K={K[k]}')

    ax1.set_ylabel('E[n], Avg Queue Size ')
    ax1.set_xlabel('% Utilization')

    ax2.set_ylabel('% Idle/Loss')
    ax2.set_xlabel('% Utilization')

    ax1.set_title(f'E[n] v Rho')
    ax2.set_title(f'P_Idle v Rho')
    ax3.set_title(f'P_Loss v Rho')

    ax1.legend()
    ax2.legend()
    ax3.legend()

  fig1.savefig(en_fig)
  fig2.savefig(idle_fig) 
  fig3.savefig(loss_fig)

  plt.show()

  return 

####################################################################################################
####################################################################################################



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

  # Rho v Metrics
  MM1_rho_results(L, C, T, timestamp)

  

  return 0 

if __name__ == '__main__':
  main()
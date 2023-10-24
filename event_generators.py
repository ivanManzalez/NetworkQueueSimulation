from rand_var_gen import rand_var_generator

def gen_arrival_t(avg_arrival,T):
  output = []
  t = 0
  # (event type, event time, data )
  while(t < T):
    dt = rand_var_generator(avg_arrival)
    output.append(("arrival", t + dt))
    t += dt

  return output

def gen_observer_t(avg_arrival, T):
  t = 0
  output =[]

  while(t < T):
    dt = rand_var_generator(avg_arrival)
    t += dt
    output.append(("observer", t))

  return output


def gen_departure_t(avg_pckt_len, trans_rate, arrivals):
  packet_lambda = 1/avg_pckt_len
  output = []

  for i in range(len(arrivals)):
    arrival_time = arrivals[i][1]
    
    # Generate random service time
    packet_length = rand_var_generator(packet_lambda)
    service_time = packet_length/trans_rate

    #if first arrival then simply departure is arrival + service time
    if i == 0:
       output.append(("departure", arrival_time + service_time))
    #but if not based on the time of arrival of the packet and the departure of previous packet calculate departure
    else:
      if arrival_time < output[i - 1][1]:
        output.append(("departure", output[i-1][1] + service_time))
      else:
        output.append(("departure",arrival_time + service_time))

  return output

def gen_service_t(arrivals, avg_pckt_len, trans_rate):
  service_times = []
  pckt_lambda = 1/avg_pckt_len
  for arrival in arrivals:
    packet_length = rand_var_generator(pckt_lambda)
    service_times.append(packet_length/trans_rate)
  return service_times


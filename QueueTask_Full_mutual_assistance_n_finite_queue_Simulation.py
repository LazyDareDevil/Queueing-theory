import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
import math 
from QueueTask_Full_mutual_assistance_n_finite_queue_Assistance import QueueModelParameters, RandomTime


class QueueModelMarkovProcessWithContinuousTime:
	def __init__(self, n: int, m: int, lamb: float, mu: float):
		self._precision = 1e-5
		self._n = n
		self._m = m
		self._lambda = lamb
		self._mu = mu
		self._number_states = self._m + self._n + 1
		self._time = 0.0
		self._memory: [(float, float)] = []
		self._iteration_index: int = 0
		self._state: int = 0
		self._memory.append((self._time, self._state))
		self._time_generator = RandomTime(self._lambda/(self._n*self._mu))

		# количество поступивших заявок
		self.count_created_requests = 0
		# среднее число заявок в системе
		self.count_all_requests = 0
		# среднее число обслуживаемых заявок
		self.number_requests_in_service = 0
		# среднее число заявок в очереди
		self.number_requests_in_queue = 0
		# среднее число занятых каналов
		self.number_channel_busy = 0
		# вероятность обслуживания заявки
		self.count_request_serviced = 0
		# вероятность канал занят
		self.count_channel_busy = 0 #???
		# вероятность система загружена полностью
		self.count_fully_busy = 0
		# среднее время занятости канала
		self.time_channel_busy = 0.0 #???
		# среднее время простоя канала
		self.time_channel_not_busy = 0.0 #???
		# среднее время неполной загрузки системы
		self.time_system_not_fully_busy = 0.0
		# среднее время полной загрузки системы
		self.time_system_fully_busy = 0.0 
		# среднеее время пребывания в очереди
		self.time_in_queue = 0.0
		# среднеее время пребывания в системе
		self.time_in_service = 0.0
		# вероятность отказа заявки
		self.count_request_rejected = 0
		# вероятность отсутствия очереди
		self.count_no_queue = 0

	def _random_value(self):
		# return random.random()
		return self._time_generator.generate()

	def _transition_probability_density(self, i: int, j: int) -> float:
		if j == i + 1:
			return self._lambda
		if j == i - 1:
			return self._n*self._mu
		return 0

	def _transition_probability_density_vector(self, i: int) -> np.array:
		result = np.zeros(self._number_states)
		for j in range(self._number_states):
			result[j] = self._transition_probability_density(i, j)
		return result
	
	def _transition_times(self, i: int) -> np.array:
		times = self._transition_probability_density_vector(i)
		generated_times = list(self._random_value())
		shape = times.shape
		for j in range(shape[0]):
			if abs(times[j]) < self._precision:
				times[j] = 1/self._precision
			else:
				# times[j] = -np.log(generated_times.pop(0))/times[j]
				times[j] = generated_times.pop(0)/times[j]
		return times

	def _next_iteration(self) -> (float, float):
		times = self._transition_times(self._state)
		min_time = np.amin(times)
		min_time_position = np.where(times == min_time)
		next_state = min_time_position[0][0]
		
		if self._state < next_state:
			self.count_created_requests += 1
		if self._state > next_state :
			self.count_request_serviced += 1
		self.count_all_requests += self._state
		self.number_requests_in_service += self._state if self._state < self._n else self._n
		if self._state >= self._n:
			if self._state == self._number_states - 1:
				self.count_request_rejected += 1
			self.time_system_fully_busy += min_time
			self.count_fully_busy += 1
		else:
			self.time_system_not_fully_busy += min_time
		if self._state > self._n:
			queue_size = self._state - self._n
			self.time_in_queue += min_time
			self.number_requests_in_queue += queue_size
		else:
			self.count_no_queue += 1
		if self._state > 0:		
			self.time_in_service += min_time
			self.number_channel_busy += self._n
			self.time_channel_busy += min_time/self._n
			self.count_channel_busy += 1
		else:
			self.time_channel_not_busy += min_time/self._n

		self._state = next_state
		self._time += min_time
		self._iteration_index += 1

		return (self._time, next_state)

	def start(self, max_time: float):
		while self._time < max_time:
			iteration_result = self._next_iteration()
			self._memory.append(iteration_result)

		self.count_all_requests /= self._iteration_index
		self.number_requests_in_service /= self._iteration_index
		self.number_requests_in_queue /= self._iteration_index
		self.number_channel_busy /= self._iteration_index
		self.count_request_serviced /= self.count_created_requests
		self.count_channel_busy /= self._iteration_index
		self.count_fully_busy /= self._iteration_index
		self.time_channel_busy /= self._time
		self.time_channel_not_busy /= self._time
		self.time_system_not_fully_busy /= self._time
		self.time_system_fully_busy /= self._time
		self.time_in_queue /= self.count_created_requests
		self.time_in_service /= self.count_created_requests
		self.count_request_rejected /= self.count_created_requests
		self.count_no_queue /= self._iteration_index

class Simualtion:
	def __init__(self, n: int, m: int, lamb: float, mu: float):
		count = 0
		self._precision = 1e-5
		self.states_results: np.array
		self.discrete_times: np.array
		self._n = n
		self._m = m
		self._lambda = lamb
		self._mu = mu
		self.count_all_requests: float = 0.0
		self.number_requests_in_service: float = 0.0
		self.number_requests_in_queue: float = 0.0
		self.number_channel_busy: float = 0.0
		self.count_request_serviced: float = 0.0
		self.count_channel_busy: float = 0.0
		self.count_fully_busy: float = 0.0
		self.time_channel_busy: float = 0.0
		self.time_channel_not_busy: float = 0.0
		self.time_system_not_fully_busy: float = 0.0
		self.time_system_fully_busy: float = 0.0
		self.time_in_queue: float = 0.0
		self.time_in_service: float = 0.0
		self.count_request_rejected: float = 0.0
		self.count_no_queue: float = 0.0

	def Calculate(self, repeats: int, general_time: float, time_delta: float):
		iteration_results = []
		iteration_times = []
		
		counter = 0
		while counter < repeats:
			queue_system = QueueModelMarkovProcessWithContinuousTime(self._n, self._m, self._lambda, self._mu)
			queue_system.start(general_time)
			iteration_results.append(queue_system._memory)
			iteration_times.append(queue_system._time/queue_system._iteration_index)
			self.count_all_requests += queue_system.count_all_requests
			self.number_requests_in_service += queue_system.number_requests_in_service
			self.number_requests_in_queue += queue_system.number_requests_in_queue
			self.number_channel_busy += queue_system.number_channel_busy
			self.count_request_serviced += queue_system.count_request_serviced
			self.count_channel_busy += queue_system.count_channel_busy
			self.count_fully_busy += queue_system.count_fully_busy
			self.time_channel_busy += queue_system.time_channel_busy
			self.time_channel_not_busy += queue_system.time_channel_not_busy
			self.time_system_not_fully_busy += queue_system.time_system_not_fully_busy
			self.time_system_fully_busy += queue_system.time_system_fully_busy
			self.time_in_queue += queue_system.time_in_queue
			self.time_in_service += queue_system.time_in_service
			self.count_request_rejected += queue_system.count_request_rejected
			self.count_no_queue += queue_system.count_no_queue
			if (counter % 500 == 0):
				print("Iteration {}".format(counter))
			counter += 1

		self.count_all_requests /= repeats
		self.number_requests_in_service /= repeats
		self.number_requests_in_queue /= repeats
		self.number_channel_busy /= repeats
		self.count_request_serviced /= repeats
		self.count_channel_busy /= repeats
		self.count_fully_busy /= repeats
		self.time_channel_busy /= repeats
		self.time_channel_not_busy /= repeats
		self.time_system_not_fully_busy /= repeats
		self.time_system_fully_busy /= repeats
		self.time_in_queue /= repeats
		self.time_in_service /= repeats
		self.count_request_rejected /= repeats
		self.count_no_queue /= repeats

		self.parameters = QueueModelParameters()
		# l с крышкой (среднее число заявок в системе)
		self.parameters.mean_request_number_in_system = self.count_all_requests
		# P_обс (вероятность обслуживания заявки)
		self.parameters.probability_serve_request = self.count_request_serviced
		# k с крышкой (среднее число занятых каналов)
		self.parameters.mean_busy_servers_number = self.number_channel_busy
		# s с крышкой (среднее число обслуживаемых заявок)
		self.parameters.mean_service_requests_number = self.number_requests_in_service
		# pi канал занят (вероятность)
		self.parameters.probability_server_is_busy = self.count_channel_busy
		# pi система загружена (вероятность)
		self.parameters.probability_system_is_fully_busy = self.count_fully_busy
		# t простоя канала (среднее время)
		self.parameters.mean_time_not_busy_server = self.time_channel_not_busy
		# t занятости канала (среднее время)
		self.parameters.mean_time_busy_server = self.time_channel_busy
		# t неполной загрузки системы (среднее время)
		self.parameters.mean_time_system_is_not_fully_busy = self.time_system_not_fully_busy
		# t полной загрузки системы (среднее время)
		self.parameters.mean_time_system_is_fully_busy = self.time_system_fully_busy
		# r с крышкой (среднее число заявок в очереди)
		self.parameters.mean_request_number_in_queue = self.number_requests_in_queue
		# t с крышкой оч (среднеее время пребывания в очереди)
		self.parameters.mean_time_request_in_queue = self.time_in_queue
		# self.parameters.mean_time_request_in_system = self.parameters.mean_time_request_in_queue + 1 / (self._n * self._mu)
		self.parameters.mean_time_request_in_system = self.time_in_service
		self.parameters.probability_no_queue = self.count_no_queue
		self.parameters.probability_request_rejection = self.count_request_rejected
		self.parameters.relative_throughput = 1 - self.parameters.probability_request_rejection
		self.parameters.absolute_bandwidth = self._lambda * self.parameters.relative_throughput
		self.parameters.check = self.parameters.check = abs(self.parameters.mean_request_number_in_system - self.parameters.mean_service_requests_number - self.parameters.mean_request_number_in_queue) < self._precision

		number_points = int(general_time/time_delta)+1
		self.discrete_times = np.linspace(0, general_time, number_points, endpoint=True)
		self.states_results = np.zeros((self._n+self._m+1, number_points), dtype=np.float32)
		counter = 0
		for counter in range(repeats):
			position = 0
			for step in range(len(iteration_results[counter])-1):
				time = iteration_results[counter][step][0]
				if time > general_time or position >= number_points:
					break
				state = iteration_results[counter][step][1]
				self.states_results[state][position] += 1
				position += 1
				while position < number_points and self.discrete_times[position] < iteration_results[counter][step+1][0]:
					self.states_results[state][position] += 1
					position += 1
			step = len(iteration_results[counter])-1
			state = iteration_results[counter][step][1]
			while position < number_points:
				self.states_results[state][position] += 1
				position += 1
		cc = np.sum(self.states_results, axis=0)
		for j in range(self.states_results.shape[1]):
			self.states_results[:, j] /= cc[j]
		self.discrete_times = self.discrete_times[:-1]
		self.states_results = self.states_results[:, :-1]

	def Draw(self):
		for i in range(self._n+1):
			plt.plot(self.discrete_times, self.states_results[i,:])
		plt.xlabel('time')
		plt.legend(["S" + str(i) for i in range(self._n+1)])
		plt.ylim([-0.1, 1])
		plt.savefig("States_before_queue.png")
		plt.clf()

		for i in range(self._n+1, self._n+self._m+1):
			plt.plot(self.discrete_times, self.states_results[i,:])	
		plt.xlabel('time')
		plt.legend(["S" + str(i) for i in range(self._n+1, self._n+self._m+1)])
		plt.ylim([0.0, 0.5])
		plt.grid()
		plt.savefig("States_after_queue.png")
		plt.clf()

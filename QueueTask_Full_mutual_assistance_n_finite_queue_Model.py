from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
from QueueTask_Full_mutual_assistance_n_finite_queue_Assistance import QueueModelParameters


def _linear_f(mat, p, t):
	return mat.dot(p)

def _runge_kutta(f, mat, t0, y0, h, num_repeats):
	y = np.zeros((num_repeats, y0.shape[0]))
	y[0] = y0
	t = t0
	for i in range(1, num_repeats):
		y_prev = y[i-1]
		k1 = f(mat, y_prev, t)
		k2 = f(mat, y_prev + (h / 2) * k1, t + h / 2)
		k3 = f(mat, y_prev + (h / 2) * k2, t + h / 2)
		k4 = f(mat, y_prev + h * k3, t + h / 2)
		y[i] = y_prev + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
		t = t + h
	return y


class QueueModel:
	def __init__(self, n: int, m: int, lamb: float, mu: float):
		self._precision = 1e-5
		self._n = n
		self._m = m
		self._lambda = lamb
		self._mu = mu
		self._number_states = self._m + self._n + 1
		self._chi = self._lambda / (self._n * self._mu)
		self._alpha = self._lambda / self._mu
		self.probabilities: np.array
		self.analytical_probabilities = np.zeros(m + n + 1)

	def _derivative_system(self) -> np.array:
		a_mat = (
			np.diag(-(self._lambda + self._n * self._mu) * np.ones(self._n + self._m + 1), 0)
			+ np.diag(self._n * self._mu * np.ones(self._n + self._m ), 1)
			+ np.diag(self._lambda * np.ones(self._n + self._m), -1)
		)
		a_mat[0][0] = -self._lambda
		a_mat[-1][-1] = -self._n * self._mu
		return a_mat

	def Solve(self, time: float, dt:float):
		num_points = int(time/dt)
		self.time = np.linspace(0, 10, num_points)
		matrix = self._derivative_system()
		print("Derivative system:\n{}".format(matrix))
		self.probabilities = np.zeros(self._n + self._m + 1)
		self.probabilities[0] = 1.0
		self.probabilities = _runge_kutta(_linear_f, matrix, self.time[0], self.probabilities, dt, num_points)
		correct = "not "
		if np.abs(1.0 - np.sum(self.probabilities[-1])) <= self._precision:
			correct = ""
		print("Solution is {}correct (sum of probabalities should be equal 1)".format(correct))

	def Draw(self):
		for i in range(self._n + 1):
			plt.plot(self.time, self.probabilities[:, i])
		plt.xlabel('time')
		plt.legend(["S" + str(i) for i in range(self._n + 1)])
		plt.ylim([-0.1, 1])
		plt.grid()
		plt.savefig("States_before_queue_model.png")
		plt.clf()

		for i in range(self._n + 1, self._n+self._m + 1):
			plt.plot(self.time, self.probabilities[:, i])
		plt.xlabel('time')
		plt.legend(["S" + str(i) for i in range(self._n + 1, self._n+self._m + 1)])
		plt.ylim([0.0, 0.5])
		plt.grid()
		plt.savefig("States_after_queue_model.png")
		plt.clf()

	def Analytical_Solution(self):
		denominator = np.sum(((self._lambda)**i/(self._n*self._mu)**i) for i in range(self._n + self._m + 1))
		self.analytical_probabilities = np.zeros(self._m + self._n + 1)
		self.analytical_probabilities[0] = 1/denominator
		for i in range(1, self._n + self._m + 1):
			self.analytical_probabilities[i] = ((self._lambda)**i/(self._n*self._mu)**i)/denominator

	def Parameters(self) -> QueueModelParameters:
		x = self._chi
		n = self._n
		m = self._m
		alpha = self._lambda/self._mu
		nm = n+m
		nm1 = nm+1
		xnm = x**(nm)
		xnm1 = x**(nm1)
		xisnotone = abs(x - 1.0) > self._precision
		result = QueueModelParameters()

		# l с крышкой (среднее число заявок в системе)
		result.mean_request_number_in_system = x* (1 - xnm * (nm*(1-x)+1)) / ((1 - xnm1)*(1 - x)) if xisnotone else (nm)/2
		# P_обс (вероятность обслуживания заявки)
		result.probability_serve_request = (1 - xnm)/(1 - xnm1) if xisnotone else (nm)/(nm1)
		# k с крышкой (среднее число занятых каналов)
		result.mean_busy_servers_number = alpha * (1 - xnm) / (1 - xnm1) if xisnotone else n*(nm)/(nm1)
		# s с крышкой (среднее число обслуживаемых заявок)
		result.mean_service_requests_number = (x / (1 - xnm1)) * ((1 - x**n *(n *(1-x) + 1)) / (1 - x) + n *(x**n - xnm)) if xisnotone else (n * (n+1)/2 + n*m) / (nm1)
		# pi канал занят (вероятность)
		result.probability_server_is_busy = x *(1- xnm)/(1-xnm1) if xisnotone else nm/nm1
		# p система загружена (вероятность)
		result.probability_system_is_fully_busy = x**n * (1 - x**(m+1)) / (1 - xnm1) if xisnotone else (m+1)/nm1
		# t простоя канала (среднее время)
		result.mean_time_not_busy_server = 1/self._lambda
		# t занятости канала (среднее время)
		result.mean_time_busy_server = result.mean_time_not_busy_server * result.probability_server_is_busy / (1 - result.probability_server_is_busy)
		# t неполной загрузки системы (среднее время)
		result.mean_time_system_is_not_fully_busy = (1 - x**n)/(x**n * (1 - x)*self._n*self._mu) if xisnotone else 1/self._mu
		# t полной загрузки системы (среднее время)
		result.mean_time_system_is_fully_busy = result.mean_time_system_is_not_fully_busy * result.probability_system_is_fully_busy / (1 - result.probability_system_is_fully_busy)
		# pn - вероятность в системе n заявок, но очередь пуста
		pn = x**n *(1 - x) / (1 - xnm1) if xisnotone else 1/nm1
		# r с крышкой (среднее число заявок в очереди)
		result.mean_request_number_in_queue = pn * x * (1 - x**m * (m *(1-x)+1)) / ((1-x)**2) if xisnotone else pn*m*(m+1) /2
		# среднее число заявок в системе равно сумме среднего числа обслуживаемых заявок и среднего числа заявок в очереди
		result.check = abs(result.mean_request_number_in_system - result.mean_service_requests_number - result.mean_request_number_in_queue) < self._precision
		# t с крышкой оч (среднеее время пребывания в очереди)
		result.mean_time_request_in_queue = result.mean_request_number_in_queue / self._lambda
		result.mean_time_request_in_system = result.mean_time_request_in_queue + 1 / (n * self._mu)
		result.probability_no_queue = sum([x**i for i in range(n)]) / sum([x**i for i in range(n+m)])
		result.probability_request_rejection = 1.0 - result.probability_serve_request
		result.relative_throughput = 1 - result.probability_request_rejection
		result.absolute_bandwidth = self._lambda * result.relative_throughput
		return result

from QueueTask_Full_mutual_assistance_n_finite_queue_Assistance import QueueModelParameters
from QueueTask_Full_mutual_assistance_n_finite_queue_Model import QueueModel
from QueueTask_Full_mutual_assistance_n_finite_queue_Simulation import Simualtion
import numpy as np
import matplotlib.pyplot as plt

def Draw_Every_State(n: int, m: int, times: np.array, res_analytic: np.array, res_runge_kutta: np.array, res_statistic: np.array):
	for i in range(0, n+m+1):
		plt.plot(times, res_statistic[i], color='m')
		plt.plot(times, res_runge_kutta[:, i], color='b')
		plt.plot(times, res_analytic[i]*np.ones(times.shape[0]), color='r')
		plt.xlabel('time')
		plt.legend(["S{} statistical".format(i), "S{} numerical (Runge-Kutta 4)".format(i), "S{} analytic (t->inf)".format(i)])
		plt.ylim([-0.1, 1.0])
		plt.savefig("S{}.png".format(i))
		plt.clf()

if __name__ == "__main__":
	repeats = 5000
	general_time = 10
	dt = 1e-3
	
	n = 5 		# number of channels
	m = 2		# places in queue
	lambd = 7 	# income flow rate
	mu = 2		# service flow rate
	# n = 2 		# number of channels
	# lambd = 4 	# income flow rate
	# mu = 6		# service flow rate
	# m = 2		# places in queue
	# n = 2 		# number of channels
	# lambd = 4 	# income flow rate
	# mu = 1		# service flow rate
	# m = 2		# places in queue
	# n = 10 		# number of channels
	# lambd = 12 	# income flow rate
	# mu = 3		# service flow rate
	# m = 5		# places in queue

	model_analytical = QueueModel(n, m, lambd, mu)
	model_analytical.Solve(general_time, dt)
	model_analytical.Analytical_Solution()
	model_analytical.Draw()
	print("Аналитическое решение при t -> inf: {}".format(model_analytical.analytical_probabilities))
	print("Численное решение Рунге-Кутты: {}".format(model_analytical.probabilities[-1]))
	print(model_analytical.Parameters().ToString())

	model = Simualtion(n, m, lambd, mu)
	model.Calculate(repeats, general_time, dt)
	print(model.parameters.ToString())
	model.Draw()

	Draw_Every_State(n, m, np.linspace(0, general_time, int(general_time/dt)), model_analytical.analytical_probabilities, model_analytical.probabilities, model.states_results)



class QueueModelParameters:
	def __init__(self):
		# l с крышкой (среднее число заявок в системе)
		self.mean_request_number_in_system : float = 0.0
		# P_обс (вероятность обслуживания заявки)
		self.probability_serve_request : float = 0.0
		# k с крышкой (среднее число занятых каналов)
		self.mean_busy_servers_number : float = 0.0
		# s с крышкой (среднее число обслуживаемых заявок)
		self.mean_service_requests_number : float = 0.0
		# pi канал занят (вероятность)
		self.probability_server_is_busy : float = 0.0
		# pi система загружена (вероятность)
		self.probability_system_is_fully_busy : float = 0.0
		# t простоя канала (среднее время)
		self.mean_time_not_busy_server : float = 0.0
		# t занятости канала (среднее время)
		self.mean_time_busy_server : float = 0.0
		# t неполной загрузки системы (среднее время)
		self.mean_time_system_is_not_fully_busy : float = 0.0
		# t полной загрузки системы (среднее время)
		self.mean_time_system_is_fully_busy : float = 0.0
		# r с крышкой (среднее число заявок в очереди)
		self.mean_request_number_in_queue : float = 0.0
		# среднее число заявок в системе == среднее число обслуживаемых заяво + среднее число заявок в очереди
		self.check : bool
		# t с крышкой оч (среднеее время пребывания в очереди)
		self.mean_time_request_in_queue : float = 0.0
		# среднеее время пребывания в системе
		self.mean_time_request_in_system : float = 0.0
		# вероятность отказа заявки
		self.probability_request_rejection: float = 0.0
		# вероятность отсутствия очереди
		self.probability_no_queue: float = 0.0
		# относительная пропускная способность
		self.relative_throughput: float = 0.0
		# абсолютная пропускная способность
		self.absolute_bandwidth: float = 0.0

	def ToString(self) -> str:
		result =""
		result += "среднее число заявок в системе: {}\n".format(self.mean_request_number_in_system)
		result += "среднее число обслуживаемых заявок: {}\n".format(self.mean_service_requests_number)
		result += "среднее число заявок в очереди: {}\n".format(self.mean_request_number_in_queue)
		tmp = "да" if self.check else "нет"
		result += "cреднее число заявок в системе == среднее число обслуживаемых заявок + среднее число заявок в очереди: {}\n".format(tmp)
		result += "среднее число занятых каналов: {}\n".format(self.mean_busy_servers_number)
		result += "вероятность обслуживания заявки: {}\n".format(self.probability_serve_request)
		result += "вероятность канал занят: {}\n".format(self.probability_server_is_busy)
		result += "вероятность система загружена полностью: {}\n".format(self.probability_system_is_fully_busy)
		result += "среднее время занятости канала: {}\n".format(self.mean_time_busy_server)
		result += "среднее время простоя канала: {}\n".format(self.mean_time_not_busy_server)
		result += "среднее время неполной загрузки системы: {}\n".format(self.mean_time_system_is_not_fully_busy)
		result += "среднее время полной загрузки системы: {}\n".format(self.mean_time_system_is_fully_busy)
		result += "среднеее время пребывания в очереди: {}\n".format(self.mean_time_request_in_queue)
		result += "среднеее время пребывания в системе: {}\n".format(self.mean_time_request_in_system)
		result += "вероятность отказа заявки: {}\n".format(self.probability_request_rejection)
		result += "вероятность отсутствия очереди: {}\n".format(self.probability_no_queue)
		result += "относительная пропускная способность: {}\n".format(self.relative_throughput)
		result += "абсолютная пропускная способность: {}\n".format(self.absolute_bandwidth)
		return result

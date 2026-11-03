from prometheus_client import start_http_server, Counter, Summary, Gauge
import requests
import time

REQUEST_COUNT = Counter('model_request_count', 'Total requests ke ML model')
REQUEST_SUCCESS = Counter('model_request_success', 'Jumlah Request berhasil')
REQUEST_FAILURE = Counter('model_request_failure', 'Jumlah Request gagal')
REQUEST_LATENCY = Summary('model_request_latency_seconds', 'Latensi request model')
INPUT_SIZE = Gauge('model_input_size', 'Jumlah item di Input')
OUTPUT_SIZE = Gauge('model_output_size', 'Jumlah prediksi dihasilkan')


MLFLOW_URL = 'http://127.0.0.1:1234/invocations'


SAMPLE_INPUT = {
    "columns": [
        "Air temperature [C]",
        "Process temperature [C]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type_L",
        "Type_M",
        "Target"
    ],
    "data": [
        [0.7492637642473495, -0.13763003020822773, 0.056036895721050625, -0.15079224533237873, -0.41823175271587815, 0.8115026712006891, -0.6509881678186371, 0],
        [-0.4509891482951158, -0.7454172097342988, -0.522465899096224, 0.34035889823143706, 1.4054263325014602, 0.8115026712006891, -0.6509881678186371, 0],
        [-0.551010224340345, 0.4026252404816005, 0.7791653892426439, -0.4214265489287674, -1.471551508832789, -1.2322818340454906, 1.5361262299910132, 0]
    ]
}

@REQUEST_LATENCY.time()
def call_model():
    REQUEST_COUNT.inc()
    INPUT_SIZE.set(len(SAMPLE_INPUT['data']))
    try:
        response = requests.post(MLFLOW_URL, json=SAMPLE_INPUT)
        if response.status_code == 200:
            REQUEST_SUCCESS.inc()
            output = response.json()
            OUTPUT_SIZE.set(len(output))
        else:
            REQUEST_FAILURE.inc()
    except Exception:
        REQUEST_FAILURE.inc()


if __name__ == "__main__":
    start_http_server(8000)
    print("Link lokasi metrik prometheus: http://localhost:8000/metrics")

    while True:
        call_model()
        time.sleep(10) 
import contextlib
import os
import signal
import subprocess

from approx_aligned_decoding.hallucination_detector.network_hallucination_detector import \
    NetworkHallucinationDetector

current_port = int(os.environ.get("PORT", 5002)) - 1


@contextlib.contextmanager
def create_hallucination_detection_server():
    global current_port
    # Weird things happen when this process is opened as a context manager
    # Better to use the finally block
    current_port += 1
    process = subprocess.Popen(['/home/ec2-user/.nvm/versions/node/v16.20.2/bin/node', '-r',
                           'ts-node/register', '--loader', 'ts-node/esm', 'src/run_server.ts'],
                               cwd="./pyright_hallucination_detector", stdout=subprocess.DEVNULL,
                               env={"PORT": str(current_port)})
    try:
        yield NetworkHallucinationDetector(f"localhost:{current_port}/detect")
    finally:
        process.send_signal(signal.SIGTERM)

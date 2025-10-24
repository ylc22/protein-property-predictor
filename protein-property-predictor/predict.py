# predict.py — Domino Endpoint Entrypoint

import os, sys
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.append(HERE)

from model import predict

def main(sequence, mode="auto", **kwargs):
    """
    This function is called automatically by Domino when your endpoint receives a request.
    Domino unpacks JSON keys (like sequence and mode) into keyword arguments.
    """
    try:
        result = predict(sequence, mode)
        return {"result": result}
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }

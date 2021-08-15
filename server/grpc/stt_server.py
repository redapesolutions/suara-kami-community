#!/usr/bin/env python3

from concurrent import futures
import os
import sys
import time
import math
import logging
import json
import grpc

import stt_service_pb2
import stt_service_pb2_grpc
from google.protobuf import duration_pb2
import numpy as np

# sys.path.append(".")
from sk import predict

import gc
gc.set_threshold(0)

class SttServiceServicer(stt_service_pb2_grpc.SttServiceServicer):
    """Provides methods that implement functionality of route guide server."""

    def get_duration(self, x):
        seconds = int(x)
        nanos = (int (x * 1000) % 1000) * 1000000
        return duration_pb2.Duration(seconds = seconds, nanos=nanos)

    def get_word_info(self, x):
        return stt_service_pb2.WordInfo(start_time=self.get_duration(x['start']),
                                        end_time=self.get_duration(x['end']),
                                        word=x['word'], confidence=x['conf'])

    def get_response(self, json_res):
        try:
            res = json.loads(json_res)
        except Exception as e:
            print(e)
        if 'partial' in res:
             alternatives = [stt_service_pb2.SpeechRecognitionAlternative(text=res['partial'])]
             chunks = [stt_service_pb2.SpeechRecognitionChunk(alternatives=alternatives, final=False)]
             return stt_service_pb2.StreamingRecognitionResponse(chunks=chunks)
        else:
             words = [self.get_word_info(x) for x in res.get('result', [])]
             confs = [w.confidence for w in words]
             if len(confs) == 0:
                 alt_conf = 0
             else:
                 alt_conf = sum(confs) / len(confs)
             alternatives = [stt_service_pb2.SpeechRecognitionAlternative(text=res['text'], words=words, confidence=alt_conf)]
             chunks = [stt_service_pb2.SpeechRecognitionChunk(alternatives=alternatives, final=True)]
             return stt_service_pb2.StreamingRecognitionResponse(chunks=chunks)

    def StreamingRecognize(self, request_iterator, context):
        request = next(request_iterator)
        partial = request.config.specification.partial_results
        buffer = []
        recent_text = ""
        repeated = 0
        recent_response = ""
        final_text = ""
        for request in request_iterator:
            buffer.append(request.audio_content)
            text,_,_,_ = predict(np.array(buffer).flatten().tobytes())
            if text:
                if recent_text == text:
                    repeated+=1
                else:
                    recent_text = text
                if repeated>8 and recent_response!=text: # repeated for 4 seconds
                    yield self.get_response(json.dumps({"text":text}))
                    recent_response = text
                    final_text += text + " "
                    repeated = 0
                    head = 40
                    buffer = [buffer[0][:head] + buffer[-1][-8000+head:]]
                elif partial:
                    yield self.get_response(json.dumps({"partial":text}))
        yield self.get_response(json.dumps({"final":final_text + text}))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor((os.cpu_count() or 1)))
    stt_service_pb2_grpc.add_SttServiceServicer_to_server(
        SttServiceServicer(), server)
    server.add_insecure_port('{}:{}'.format("0.0.0.0", 5001))
    server.start()
    print("started: localhost:5001")
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()

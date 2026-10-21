from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response_data = {
            "status": "online",
            "project": "PullUp AI Kinematics & SOTA Benchmark",
            "author": "Autumn-Crystals",
            "models_benchmarked": [
                "ViTPose+ Large",
                "ViTPose Base",
                "YOLOv8-Pose",
                "MediaPipe BlazePose"
            ],
            "metrics": {
                "rep_precision": "100.0%",
                "rep_recall": "100.0%",
                "max_map": 81.1
            }
        }
        self.wfile.write(json.dumps(response_data, indent=2).encode('utf-8'))
        return

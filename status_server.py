# Save as status_server.py
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import subprocess
from datetime import datetime

class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Get service status
            service_active = subprocess.run(['systemctl', 'is-active', 'cctv-detection.service'], 
                                          capture_output=True, text=True).stdout.strip() == 'active'
            
            # Get recent clips
            clips_dir = '/home/nemes1s/Desktop/cctv-detection/prototype-1/output_clips'
            recent_clips = []
            if os.path.exists(clips_dir):
                clips = sorted([f for f in os.listdir(clips_dir) if f.endswith('.mp4')])[-10:]
                recent_clips = clips
            
            # Get system stats
            uptime = subprocess.run(['uptime'], capture_output=True, text=True).stdout.strip()
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'service_active': service_active,
                'recent_clips': recent_clips,
                'total_clips': len(os.listdir(clips_dir)) if os.path.exists(clips_dir) else 0,
                'uptime': uptime
            }
            
            # Return JSON response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(status, indent=2).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode())

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8080), StatusHandler)
    print("Status server running on http://localhost:8080")
    server.serve_forever()
